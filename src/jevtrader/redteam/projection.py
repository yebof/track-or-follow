"""Free dry runs and cost projections for the post-unlock paid runs of the red team and FAITH (pre-registration draft
§13 item 2: "the projection comes from a free dry run: the exact requests go to a fake client calibrated on measured
per-request costs"; CLAUDE.md budget rule: print the projection against the sub-cap before any paid run, abort if
over; no paid smoke test).

- `ProjectingEmulator`: the paid emulator's class (`llm-logprob`, `emu-v1`, pinned snapshot) with its request keys,
  `cfg_hash` and bank hash unchanged, so response-cache hits are exactly the paid run's; but `_ask` never reaches a
  client: it answers GATES OPEN (every screening, guard and gate question is answered so that the item goes on:
  N0 keeps every sentence, P(about_focal) = P(new_development) = .9, P(instructs_reader) = P(promotional) = .05, N2 says
  "new"; everything else a deterministic hash distribution), so every conditional request the paid run could send is
  sent: an upper bound on the request count. Usage comes from the token calibration (`results/redteam/
  token_calibration.json`, fitted on MAIN's billed prompt tokens: `experiments/redteam/calibrate_tokens.py`), cost from
  the archived price table (`s1.pricing`). Modes: `high` (the calibration's p95 bytes-to-tokens ratio, p90 output
  tokens per question, no provider prompt caching) and `point` (the fit, the median output, MAIN's measured cached
  share). The ABORT decision uses `high`.
- `ReadThroughCache`: the real response cache opened READ-ONLY (sqlite `mode=ro`) plus an in-memory layer for the dry
  run's own answers, so a dry run sees the paid run's hits and never writes the real cache.
- `ReaskProxy`: FAITH's null arm (design_v1 §10.3, validity M8: "re-ask the unedited state"). Requests of the named
  programs go to a second instance of the same backend WITHOUT a response cache (same request key, same prompt; the
  answer is neither read from nor written to the cache); everything else goes to the cached backend. A re-ask with a
  throwaway field in the state would change the prompt; this re-asks the identical prompt.
- `preflight`: the printed line and the abort rule: projection (high) against the run's sub-cap headroom (sub-cap minus
  the ledger's spend under that run tag) and the hard cap; abort (exit 3) if over. The ledger is read-only here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from jevtrader.s1.backends import LogprobEmulator, _duplicate
from jevtrader.s1.types import Answer, S1Response, canonical_json, labels, question_to_json

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = ROOT / "results" / "redteam" / "token_calibration.json"
CONFIG = ROOT / "configs" / "backends.yaml"


# ------------------------------------------------------------------------------------------------ calibration
@dataclass(frozen=True)
class Calibration:
    n0: tuple[float, float, float]  # (tokens per byte, intercept, p95 ratio)
    other: tuple[float, float, float]
    out_median: float  # output tokens per question
    out_p90: float
    cached_share: float  # provider prompt-cache share of input tokens (point mode)
    source: str

    @classmethod
    def load(cls, path: Path = CALIBRATION) -> Calibration:
        """The fitted calibration, or the conservative default (`backends.estimate_tokens`' 3.5 bytes per token,
        no caching, 4 output tokens per question) when no calibration file exists."""
        p = Path(path)
        if not p.exists():
            r = 1 / 3.5
            return cls((r, 0.0, r), (r, 0.0, r), 4.0, 4.0, 0.0, "default (estimate_tokens: 3.5 bytes per token)")
        c = json.loads(p.read_text())
        f0, f1 = c["input"]["N0"], c["input"]["N1"]
        oq = c["output_per_question"]
        n1 = oq.get("N1", {"median": 3.0, "p90": 4.0, "cached_share": 0.0})
        tot_in = sum(v.get("n", 0) for v in oq.values()) or 1
        share = sum(v["cached_share"] * v.get("n", 0) for v in oq.values()) / tot_in
        return cls((f0["tokens_per_byte"], f0["intercept"], f0["ratio_p95"]),
                   (f1["tokens_per_byte"], f1["intercept"], f1["ratio_p95"]),
                   max(v["median"] for k, v in oq.items() if k in ("N0", "N1", "N2", "N3", "T")) if oq else
                   n1["median"], max(v["p90"] for k, v in oq.items() if k in ("N0", "N1", "N2", "N3", "T")),
                   float(share), str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))

    def tokens(self, nbytes: int, n0: bool, high: bool) -> int:
        a, b, p95 = self.n0 if n0 else self.other
        return int(math.ceil(p95 * nbytes if high else max(a * nbytes + b, 1.0)))

    def to_json(self) -> dict:
        return {"n0": list(self.n0), "other": list(self.other), "out_median": self.out_median,
                "out_p90": self.out_p90, "cached_share": self.cached_share, "source": self.source}


# ------------------------------------------------------------------------------------------------ cache
class ReadThroughCache:
    """The real response cache, READ-ONLY, plus an in-memory layer (module doc). Duck-types `ResponseCache`."""

    def __init__(self, path: Path | None):
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.path = Path(path) if path is not None else None
        self._db = None
        if self.path is not None and self.path.exists():
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False, timeout=60)
        self.real_hits = 0
        self.mem_hits = 0

    def get(self, key: str) -> S1Response | None:
        with self._lock:
            rec = self._mem.get(key)
            if rec is not None:
                self.mem_hits += 1
            elif self._db is not None:
                row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    rec = json.loads(row[0])
                    self.real_hits += 1
        if rec is None:
            return None
        r = S1Response.from_json(rec)
        r.cached, r.request_key = True, key
        return r

    def put(self, key: str, response: S1Response) -> None:
        if response.degraded or any(a.probs is None for a in response.answers.values()):
            raise ValueError("degraded responses are never cached")
        with self._lock:
            self._mem[key] = response.to_json()

    async def aget(self, key: str) -> S1Response | None:
        return await asyncio.to_thread(self.get, key)

    async def aput(self, key: str, response: S1Response) -> None:
        self.put(key, response)

    def __len__(self) -> int:
        return len(self._mem)


# ------------------------------------------------------------------------------------------------ the fake emulator
GATES_OPEN: dict[str, float] = {  # P(yes) of the screening / guard / gate Nouls (module doc)
    "about_focal": 0.9, "new_development": 0.9, "instructs_reader": 0.05, "promotional": 0.05, "important": 0.8,
    "unconfirmed": 0.1, "serious_problem": 0.05, "problem_unresolved": 0.5, "scheduled": 0.5, "continuing_story": 0.5,
}


def gates_open_answer(name: str, q, state) -> dict[str, float]:
    labs = labels(q)
    if q.kind == "noul":
        if name.startswith("reports_"):
            p = 0.9
        elif name.startswith("judges_"):
            p = 0.1
        elif name.startswith("same_as_prior_"):
            p = 0.1
        else:
            p = GATES_OPEN.get(name)
        if p is not None:
            return {"yes": p, "no": 1 - p}
    h = hashlib.sha256(canonical_json([state, question_to_json(q)]).encode()).digest()
    z = [3.0 * (h[i % len(h)] / 255.0 - 0.5) * 2 for i in range(len(labs))]
    e = [math.exp(v) for v in z]
    return {lab: v / sum(e) for lab, v in zip(labs, e)}


class _NoClient:
    def __getattr__(self, k):
        raise RuntimeError("a projecting emulator never calls a provider")


class ProjectingEmulator(LogprobEmulator):
    """The paid emulator's identity with a calibrated fake in place of the provider (module doc)."""

    def __init__(self, *, calibration: Calibration | None = None, mode: str = "high", **kw):
        if mode not in ("high", "point"):
            raise ValueError("mode: high | point")
        kw.pop("client", None)
        kw["prompt_cache"] = False
        super().__init__(client=_NoClient(), **kw)
        self.calibration = calibration or Calibration.load()
        self.mode = mode
        self.fake_calls = 0

    async def _ask(self, state, questions):
        self.fake_calls += 1
        msgs, maps = self.messages(state, questions)
        nbytes = sum(len(m["content"].encode("utf-8")) for m in msgs)
        n0 = all(n.startswith(("reports_", "judges_")) for n in questions)
        high = self.mode == "high"
        cal = self.calibration
        n_in = cal.tokens(nbytes, n0, high)
        nq = len(questions)
        n_out = min(8 * nq + 16, int(math.ceil((cal.out_p90 if high else cal.out_median) * nq)) + 1)
        cached = 0 if high else int(cal.cached_share * n_in)
        answers = {name: Answer(q.kind, gates_open_answer(name, q, state)) for name, q in questions.items()}
        await asyncio.sleep(0)
        return S1Response(answers=answers, model=self.model, backend=self.backend_id, input_tokens=n_in,
                          output_tokens=n_out, cached_tokens=cached, response_model=self.model, degraded=[],
                          meta={"projected": self.mode})


def projecting_backend(mode: str = "high", cache_path: Path | None = None, calibration: Calibration | None = None,
                       cache: bool = True) -> ProjectingEmulator:
    """A dry-run backend: the paid emulator's identity from configs/backends.yaml, the real cache read-only (or no
    cache), no ledger, no client."""
    from jevtrader.s1.backends import _bank_sha256, load_backend_config

    cfg = load_backend_config(CONFIG)
    kw = dict(cfg["backends"]["llm-logprob"])
    for k in ("prompt_cache", "rpm", "max_concurrency"):
        kw.pop(k, None)
    path = cache_path if cache_path is not None else ROOT / cfg["cache"]["path"]
    return ProjectingEmulator(calibration=calibration, mode=mode, cache=ReadThroughCache(path) if cache else None,
                              ledger=None, bank_hash=_bank_sha256(), max_concurrency=64, **kw)


# ------------------------------------------------------------------------------------------------ null arm
class ReaskProxy:
    """FAITH's null arm (module doc): requests whose program (the ledger tag's last segment) is in `programs` go to
    `uncached` (the same backend class without a response cache), the rest to `cached`. Tags get `suffix`."""

    def __init__(self, cached, uncached, programs: Sequence[str] = ("N3",), suffix: str = "null"):
        if getattr(uncached, "cache", None) is not None:
            raise ValueError("the re-ask backend must have no response cache")
        if (cached.backend_id, cached.model, cached.prompt_version, cached.cfg_hash) != (
                uncached.backend_id, uncached.model, uncached.prompt_version, uncached.cfg_hash):
            raise ValueError("the re-ask backend must be the same backend (id, model, prompt, cfg_hash)")
        self._c, self._u, self.programs, self.suffix = cached, uncached, tuple(programs), suffix
        self.reasked = 0

    def __getattr__(self, k):
        return getattr(self._c, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        prog = tag.rsplit("/", 1)[-1] if tag else ""
        if prog in self.programs:
            self.reasked += 1
            head = tag.rsplit("/", 1)[0] if "/" in tag else ""
            r = await self._u.ask(state, questions, tag=f"{head}-{self.suffix}/{prog}" if head else tag, **kw)
            r = _duplicate(r, r.request_key) if r.cached else r
            r.meta = {**r.meta, "reask": True}
            return r
        return await self._c.ask(state, questions, tag=tag, **kw)


# ------------------------------------------------------------------------------------------------ accounting
def spend_of(requests: pd.DataFrame) -> dict[str, Any]:
    """Projected USD of a dry run's requests table (uncached rows; a coalesced duplicate is `cached`), by program."""
    if requests is None or not len(requests):
        return {"usd": 0.0, "requests": 0, "paid": 0, "by_program": {}}
    paid = requests[~requests["cached"].astype(bool)]
    by = paid.groupby("program")["usd"].agg(["size", "sum"])
    return {"usd": round(float(paid["usd"].sum()), 4), "requests": int(len(requests)), "paid": int(len(paid)),
            "by_program": {k: {"paid": int(r["size"]), "usd": round(float(r["sum"]), 4)} for k, r in by.iterrows()}}


def ledger_state(run: str, config: Path = CONFIG, root: Path = ROOT) -> dict:
    """Sub-cap, hard cap and the ledger's spend (total and under `run`), read-only."""
    import yaml

    cfg = yaml.safe_load(Path(config).read_text())["ledger"]
    path = root / cfg["path"]
    total = spent = 0.0
    if path.exists():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for tag, usd in con.execute("SELECT tag, COALESCE(SUM(usd), 0) FROM spend GROUP BY tag"):
                total += float(usd)
                if (tag or "").split("/", 1)[0] == run:
                    spent += float(usd)
        finally:
            con.close()
    caps = cfg.get("sub_caps_usd") or {}
    return {"run": run, "sub_cap": None if run not in caps else float(caps[run]), "run_spent": round(spent, 4),
            "ledger_total": round(total, 4), "hard_cap": float(cfg["hard_cap_usd"])}


@dataclass
class Preflight:
    run: str
    label: str
    projected_high: float
    projected_point: float
    sub_cap: float | None
    run_spent: float
    committed: float  # projections of stages of the same run already approved in this process, not yet spent
    ledger_total: float
    hard_cap: float

    @property
    def headroom(self) -> float:
        return 0.0 if self.sub_cap is None else self.sub_cap - self.run_spent - self.committed

    @property
    def ok(self) -> bool:
        return (self.sub_cap is not None and self.sub_cap > 0 and self.projected_high <= self.headroom
                and self.ledger_total + self.committed + self.projected_high <= self.hard_cap)

    def line(self) -> str:
        cap = "none (refused)" if self.sub_cap is None else f"${self.sub_cap:.2f}"
        return (f"PREFLIGHT {self.run} [{self.label}]: projected ${self.projected_high:.2f} high "
                f"(point ${self.projected_point:.2f}) vs sub-cap {cap} (run spent ${self.run_spent:.2f}, committed "
                f"${self.committed:.2f}, headroom ${self.headroom:.2f}); ledger ${self.ledger_total:.2f} of hard cap "
                f"${self.hard_cap:.2f} -> {'OK' if self.ok else 'ABORT'}")

    def to_json(self) -> dict:
        return {**self.__dict__, "headroom": round(self.headroom, 4), "ok": self.ok}


def preflight(run: str, label: str, high: float, point: float, committed: float = 0.0,
              config: Path = CONFIG, root: Path = ROOT) -> Preflight:
    s = ledger_state(run, config, root)
    return Preflight(run, label, round(high, 4), round(point, 4), s["sub_cap"], s["run_spent"], round(committed, 4),
                     s["ledger_total"], s["hard_cap"])


def cut_from_end(arms: Sequence[tuple[str, float]], headroom: float) -> tuple[list[str], list[str]]:
    """FAITH's rule (pre-registration draft §9): arms in their registered order; those that do not fit are cut from
    the END of the list, i.e. the longest prefix whose cumulative projection fits the headroom runs."""
    kept, cum = [], 0.0
    for i, (name, usd) in enumerate(arms):
        if cum + usd > headroom:
            return kept, [n for n, _ in arms[i:]]
        cum += usd
        kept.append(name)
    return kept, []


class SharedGuard:
    """The in-process guard of a paid stage (`experiments/dev/run_dev.SpendGuard` semantics) shared by several
    backends (FAITH's cached and uncached instances): before a call it reserves the backend's `estimate_cost` and
    refuses the call if spent + reserved would pass `max_usd` (the headroom the preflight approved); after the call the
    reservation becomes the actual cost. The ledger's sub-cap check stays in force underneath."""

    def __init__(self, max_usd: float):
        self.max_usd, self.spent, self.reserved, self.calls, self.refused = float(max_usd), 0.0, 0.0, 0, 0

    def wrap(self, s1) -> _Guarded:
        return _Guarded(s1, self)


class _Guarded:
    def __init__(self, s1, guard: SharedGuard):
        self._s1, self._g = s1, guard

    def __getattr__(self, k):
        return getattr(self._s1, k)

    @property
    def spent(self) -> float:
        return self._g.spent

    async def ask(self, state, questions, tag: str = "", **kw):
        from jevtrader.s1.budget import BudgetExceeded

        g = self._g
        est = max(0.0, float(self._s1.estimate_cost(state, questions))) if hasattr(self._s1, "estimate_cost") else 0.0
        if g.spent + g.reserved + est > g.max_usd:
            g.refused += 1
            raise BudgetExceeded(f"stage guard: spent ${g.spent:.4f} + reserved ${g.reserved:.4f} + this call "
                                 f"${est:.4f} > ${g.max_usd:.4f}")
        g.reserved += est
        try:
            resp = await self._s1.ask(state, questions, tag=tag, **kw)
        finally:
            g.reserved -= est
        if not resp.cached:
            g.spent += float(resp.cost_usd)
            g.calls += 1
        return resp


def paid_guard(s1, max_usd: float):
    """One backend under a `SharedGuard` of its own."""
    return SharedGuard(max_usd).wrap(s1)


def config_ledger(config: Path = CONFIG, root: Path = ROOT):
    """The spend ledger of configs/backends.yaml (hard cap and sub-caps as configured; never changed here)."""
    import yaml

    from jevtrader.s1.budget import SpendLedger

    lc = yaml.safe_load(Path(config).read_text())["ledger"]
    return SpendLedger(root / lc["path"], hard_cap_usd=float(lc.get("hard_cap_usd", 245.0)),
                       sub_caps={k: float(v) for k, v in (lc.get("sub_caps_usd") or {}).items()},
                       batch_size=int(lc.get("batch_size", 32)))


def frames_spend(out_dirs: Sequence[Path]) -> dict:
    reqs = [pd.read_parquet(Path(d) / "requests.parquet") for d in out_dirs if (Path(d) / "requests.parquet").exists()]
    return spend_of(pd.concat(reqs, ignore_index=True) if reqs else None)


def mapping_sum(d: Mapping[str, float]) -> float:
    return float(sum(d.values()))
