"""The unblinded arm of PRE (design_v1 §13.2 PRE, §13.7 D3): the same Layer-A run with firm identities revealed.

D3 compares the typed system on blinded and unblinded states: Delta_P = IC_unblinded - IC_blinded per period and
DiD = Delta_pre - Delta_clean, glasserman2023assessing's decomposition (original vs anonymized text, p.2, p.5, p.15).
To make the two arms differ *only* in what the model is told about identity, the unblinded arm runs the ordinary
blinded pipeline (the same D0, blinder, screening, clustering, R3 checks, write gate and memory, all on blinded text)
and reveals identity at the backend boundary, just before a request is sent:

- every study-firm code name `Company-XXXXXXXX` in the state *and* in the question texts (N2 carries prior events in
  its instructions) is replaced by that firm's SEC registrant name valid at the decision date d_t
  (`programs.probes.company_name`: the name the D1/LAP recall probes use, cleaned as the blinder cleans it; the ticker
  when no SEC name is archived);
- in a request where at least one name was revealed, the state note's clause that names are replaced by code names
  is removed (bank `conventions.note_text` -> `UNBLINDED_NOTE`), so the prompt does not tell the model that the names
  it sees are codes. A request that carries no study-firm code name (M, most P1 items, many N0 sentence lists) is left
  byte-identical, so both arms send it with the same key and receive the same answer.

What stays blinded in both arms (the contrast is firm identity only): relative times and `[date]` masks, `[level]`
masks, `[executive]` person masks, and every state-schema rule. A code name that is not a study firm's (the bank
note's example, the content-free probe's reserved code) is left as it is.

The decision week comes from the unit being run: `WeekScoped` wraps each program and sets `CURRENT_WEEK` to ctx.t
around `run`, so every request a unit sends (in its own task and the tasks it spawns) is unblinded with names valid at
that unit's week. `UnblindingBackend.ask` refuses a request sent outside a unit (no silent mix of weeks).

Request keys are computed by the backend on the unblinded (state, questions), so the two arms share a cached answer
only for a request that carries no identity at all, which is then byte-identical in both arms. `requests.parquet.state_sha256` keeps the hash of the blinded
state the program built (the unblinded state is a function of it and of d_t).
"""

from __future__ import annotations

import contextvars
import dataclasses
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

VERSION = "unblind-names-v1"  # enters the unblinded run's variant set and context description (resume safety)
CODE_NAME = re.compile(r"Company-[0-9A-F]{8}")
UNBLINDED_NOTE = "Weekly review for a US stock portfolio. Times are given relative to the review."
CURRENT_WEEK: contextvars.ContextVar[pd.Timestamp | None] = contextvars.ContextVar("pre_unblind_week", default=None)


class UnscopedRequest(RuntimeError):
    """A request reached the unblinding backend outside a `WeekScoped` unit (the decision week is unknown)."""


def blinded_notes(bank) -> frozenset[str]:
    """The bank's note texts that describe the code names (conventions.note_text*), replaced in the unblinded arm."""
    conv = bank.conventions
    return frozenset(v for k, v in conv.items() if k.startswith("note_text") and isinstance(v, str)
                     and "code names" in v)


class Unblinder:
    """Replaces study-firm code names by names valid at a decision date (`name_of(symbol, t)`), and, in a request where
    a name was revealed, the code-name note by `UNBLINDED_NOTE`. Thread-safe counters record what was replaced."""

    def __init__(self, code_to_symbol: Mapping[str, str], name_of: Callable[[str, pd.Timestamp], str],
                 notes: frozenset[str] = frozenset()):
        self.code_to_symbol = dict(code_to_symbol)
        self._name_of = name_of
        self.notes = frozenset(notes)
        self._names: dict[tuple[str, pd.Timestamp], str] = {}
        self._lock = threading.Lock()
        self.replaced: Counter = Counter()  # symbol -> number of code-name occurrences replaced
        self.unknown: Counter = Counter()  # code names left as they are (not a study firm)
        self.notes_replaced = 0
        self.requests = 0
        self.requests_revealed = 0  # requests in which at least one name was revealed
        self.requests_unchanged = 0  # requests sent byte-identical to the blinded arm

    def name(self, symbol: str, t: pd.Timestamp) -> str:
        key = (symbol, pd.Timestamp(t))
        with self._lock:
            got = self._names.get(key)
        if got is None:
            got = str(self._name_of(symbol, pd.Timestamp(t))) or symbol
            if CODE_NAME.search(got):
                raise ValueError(f"name of {symbol} at {t} looks like a code name: {got!r}")
            with self._lock:
                self._names[key] = got
        return got

    def text(self, s: str, t: pd.Timestamp, hits: list[int] | None = None) -> str:
        def rep(m: re.Match) -> str:
            sym = self.code_to_symbol.get(m.group(0))
            if sym is None:
                with self._lock:
                    self.unknown[m.group(0)] += 1
                return m.group(0)
            with self._lock:
                self.replaced[sym] += 1
            if hits is not None:
                hits[0] += 1
            return self.name(sym, t)

        return CODE_NAME.sub(rep, s)

    def value(self, x: Any, t: pd.Timestamp, hits: list[int] | None = None) -> Any:
        """A JSON value with every string (dict keys included) unblinded; containers keep their order and type.
        `hits[0]` counts the names revealed."""
        if isinstance(x, str):
            return self.text(x, t, hits)
        if isinstance(x, dict):
            return {self.text(k, t, hits) if isinstance(k, str) else k: self.value(v, t, hits) for k, v in x.items()}
        if isinstance(x, list):
            return [self.value(v, t, hits) for v in x]
        if isinstance(x, tuple):
            return tuple(self.value(v, t, hits) for v in x)
        return x

    def question(self, q, t: pd.Timestamp, hits: list[int] | None = None):
        """A copy of a Noul/Choice/Score with its instructions, criteria, option texts and level texts unblinded
        (option labels, the answer keys, never contain code names and are kept)."""
        changes: dict[str, Any] = {"instructions": self.value(q.instructions, t, hits)}
        if q.kind == "noul":
            changes["true"] = self.value(q.true, t, hits)
            changes["false"] = self.value(q.false, t, hits)
        elif q.kind == "choice":
            changes["options"] = {k: self.value(v, t, hits) for k, v in q.options.items()}
        else:
            changes["levels"] = tuple(self.value(v, t, hits) for v in q.levels)
        return dataclasses.replace(q, **changes)

    def request(self, state: Any, questions: Mapping[str, Any], t: pd.Timestamp) -> tuple[Any, dict[str, Any]]:
        """The (state, questions) the unblinded arm sends: names revealed in both; the note replaced only when a name
        was revealed; the originals returned unchanged (same objects) when nothing was revealed."""
        hits = [0]
        st = self.value(state, t, hits)
        qs = {n: self.question(q, t, hits) for n, q in questions.items()}
        with self._lock:
            self.requests += 1
            if not hits[0]:
                self.requests_unchanged += 1
                return state, dict(questions)
            self.requests_revealed += 1
        if isinstance(st, dict) and st.get("note") in self.notes:
            with self._lock:
                self.notes_replaced += 1
            st = {**st, "note": UNBLINDED_NOTE}  # key order kept (the note stays first)
        return st, qs

    def state(self, state: Any, t: pd.Timestamp) -> Any:
        """The state alone, as `request` sends it with no question (tests and the arm checks)."""
        return self.request(state, {}, t)[0]

    def report(self) -> dict:
        with self._lock:
            return {"version": VERSION, "requests": self.requests, "requests_revealed": self.requests_revealed,
                    "requests_unchanged": self.requests_unchanged, "notes_replaced": self.notes_replaced,
                    "code_names_replaced": int(sum(self.replaced.values())), "firms_named": len(self.replaced),
                    "unknown_code_names": dict(self.unknown), "names": {f"{s}@{t.date()}": n for (s, t), n in
                                                                        sorted(self._names.items())}}


class UnblindingBackend:
    """Backend proxy: unblinds (state, questions) at the decision week of the calling unit, then asks the wrapped
    backend (which computes the request key on what is actually sent). Every other attribute is the backend's."""

    def __init__(self, s1, unblinder: Unblinder, on_send: Callable[[Any, Mapping, str, Any], None] | None = None):
        self._s1, self.unblinder, self._on_send = s1, unblinder, on_send

    def __getattr__(self, k):
        return getattr(self._s1, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        t = CURRENT_WEEK.get()
        if t is None:
            raise UnscopedRequest(f"request {tag!r} sent outside a week-scoped unit: the decision week is unknown")
        st, qs = self.unblinder.request(state, questions, t)
        resp = await self._s1.ask(st, qs, tag=tag, **kw)
        if self._on_send is not None:
            self._on_send(st, qs, tag, resp)
        return resp


class WeekScoped:
    """Program proxy: runs the wrapped program with `CURRENT_WEEK` set to the unit's decision date. Name, version,
    taint, dependencies and every other attribute are the wrapped program's (so the runner's fingerprints, ordering
    and memory rights are unchanged)."""

    def __init__(self, program):
        self._p = program

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return getattr(self._p, k)

    async def run(self, ctx, s1):
        token = CURRENT_WEEK.set(pd.Timestamp(ctx.t))
        try:
            return await self._p.run(ctx, s1)
        finally:
            CURRENT_WEEK.reset(token)


def week_scoped(names, resolve) -> dict[str, WeekScoped]:
    """`GeneratorSpec.program_objects` for the unblinded arm: every program of the run, week-scoped."""
    return {n: WeekScoped(resolve(n)) for n in names}
