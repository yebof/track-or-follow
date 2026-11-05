"""Content-free states: the prior-lean probe VC-3 (design_v2_delta §3.3, bank patch QB-2, `state_schemas.CF`).

For each program X in `state_schemas.CF.applies_to` (N0, N1, N2, N3, T, FA, FB, P1, P2, M, V, H, HS, D) and each v in
`conventions.content_free.values` ("N/A", "", "[not provided]"): X's state with the note kept, `focal_company` =
{code_name: `conventions.content_free.code_name` (Company-00000000), industry: v, size: v}, every other leaf set to v
and every list set to [] -- except the lists X never sends empty (N0 sentences, N3 this_week_events, V evidence,
H evidence.news_items, HS headlines), which keep one element whose leaves are v. Instruction fields that carry state
(N2 `prior_event`, V `claim`) are set to v as well; the industry texts of the `_k` questions are question text and
stay. All of X's questions are asked jointly, templates expanded once (`_i`, `_j`, HS `_k`: index 0; P1 and M `_k`:
the 12 industries). D's questions each have their own state (one request per question); `reid_company`'s five name
options are neutral labels.

The answers measure the lean of each question with no content: diagnostic only, never an input to decisions or
learning (`conventions.content_free.rule`). They are written under program label "CF" with qid '<X>.<question>'
(`Request.build(relabel=...)`), so they can never be read as X's own answers. CF states are built here, never through
`render.envelope`: the R3 vocabulary check does not apply to them (`state_schemas.CF.code`).

`cf_state(program, v)` builds from the schema skeletons below (key sets mirror the bank schemas and the state
builders; a test checks them against real T, FA, FB, P1, P2 and M states); `cf_from_template(program, state, v)`
turns any real state of X into its content-free version, for programs whose builders live elsewhere.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pandas as pd

from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import ProgramOutput, QSpec, Request, WeekContext, ask
from jevtrader.programs.fundamentals import FUNDAMENTAL_FIELDS, VALUATION_FIELDS
from jevtrader.programs.macro import CALENDAR_KEY, INDICATOR_KEYS

MARKET = "MARKET"
class _Leaf:
    """Placeholder for a leaf in the skeletons (survives deepcopy as the same object)."""

    def __deepcopy__(self, memo):
        return self

    def __repr__(self) -> str:
        return "LEAF"


LEAF = _Leaf()
PRICE_ACTION_KEYS = ("past week return", "past month return", "past 3-month return", "12-1 month momentum", "RSI(14)",
                     "price vs 50-day average", "price vs 200-day average", "drawdown from 52-week high",
                     "volatility (60d, annualized)", "recent volume vs normal", "3-month performance vs sector")
# lists a program never sends empty keep one element (dotted path within the state)
REQUIRED_ONE = {"N0": {"sentences"}, "N3": {"this_week_events"}, "V": {"evidence"}, "H": {"evidence.news_items"},
                "HS": {"headlines"}}
_ITEM = {"headline": LEAF, "summary": LEAF, "source": LEAF, "published": LEAF, "companies_tagged": []}
_EVENT = {"event": LEAF, "effect": LEAF, "important": LEAF, "confirmed": LEAF, "new": LEAF, "outlets": LEAF,
          "vs_expectations": LEAF}
_FOCAL = "focal_company"
SKELETONS: dict[str, dict[str, Any]] = {
    "N0": {"sentences": [LEAF]},
    "N1": {_FOCAL: LEAF, "item": _ITEM},
    "N2": {_FOCAL: LEAF, "item": _ITEM},
    "N3": {_FOCAL: LEAF, "this_week_events": [_EVENT], "earlier_events": [], "stock_reaction": LEAF},
    "T": {_FOCAL: LEAF, "price_action": dict.fromkeys(PRICE_ACTION_KEYS, LEAF),
          "market": {"broad US market over the past month": LEAF, "market regime": LEAF}},
    "FA": {_FOCAL: LEAF, "business_excerpts": []},
    "FB": {_FOCAL: LEAF, "fundamentals": {k: LEAF for k, _, _ in FUNDAMENTAL_FIELDS},
           "valuation": {k: LEAF for k, _ in VALUATION_FIELDS}},
    "P1": {"item": {"headline": LEAF, "summary": LEAF, "published": LEAF}},
    "P2": {"policy_themes_this_week": [], "policy_themes_prior_weeks": [],
           "uncertainty_indices": {"economic policy uncertainty (30-day average)": LEAF,
                                   "geopolitical risk (30-day average)": LEAF}},
    "M": {"indicators": dict.fromkeys(INDICATOR_KEYS, LEAF), "calendar": {CALENDAR_KEY: LEAF}},
    "V": {_FOCAL: LEAF, "evidence": [{"id": LEAF, "text": LEAF}]},
    "H": {_FOCAL: LEAF, "evidence": {"news_items": [{"headline": LEAF, "summary": LEAF, "published": LEAF}]}},
    "HS": {"headlines": [LEAF]},
}
# D: one state per question (bank state_schemas.D)
D_STATES: dict[str, Any] = {
    "reid_year": ("N1", None),  # a (content-free) N1 state
    "reid_company": ("T", None),  # a (content-free) T state
    "recall_close_higher": (None, {"company_name": LEAF, "earlier_date": LEAF, "later_date": LEAF}),
    "recall_beats_median": (None, {"company_name": LEAF, "period": {"start": LEAF, "end": LEAF}}),
    "recall_market_up": (None, {"period": {"start": LEAF, "end": LEAF}}),
    "recall_direction": (None, {"company_name": LEAF, "period": {"start": LEAF, "end": LEAF}}),
}
REID_LABELS = ("option_1", "option_2", "option_3", "option_4", "option_5")


def _fill(x: Any, v: str, required: set[str], path: str = "") -> Any:
    if x is LEAF or isinstance(x, (str, int, float)) or x is None:
        return v
    if isinstance(x, dict):
        return {k: _fill(val, v, required, f"{path}.{k}" if path else k) for k, val in x.items()}
    if isinstance(x, (list, tuple)):
        return [_fill(x[0], v, required, path)] if path in required and len(x) else []
    raise TypeError(f"unexpected state value {type(x).__name__}")


def _focal(bank: Bank, v: str) -> dict[str, str]:
    return {"code_name": bank.conventions["content_free"]["code_name"], "industry": v, "size": v}


def cf_from_template(bank: Bank, program: str, state: dict, v: str) -> dict:
    """The content-free version of a real state of `program` (keys kept, see the module doc)."""
    req = REQUIRED_ONE.get(program, set())
    out = {}
    for k, val in state.items():
        if k == "note":
            out[k] = val
        elif k == _FOCAL:
            out[k] = _focal(bank, v)
        else:
            out[k] = _fill(val, v, req, k)
    return out


def cf_state(bank: Bank, program: str, v: str) -> dict:
    """The content-free state of `program` from its skeleton (note first, then the schema's field order)."""
    if program not in SKELETONS:
        raise KeyError(f"no content-free skeleton for {program!r}")
    sk = SKELETONS[program]
    state: dict[str, Any] = {"note": bank.note(program)}
    for k in bank.state_fields(program):
        if k == "note":
            continue
        if k not in sk:
            raise KeyError(f"{program}: schema field {k!r} has no skeleton")
        state[k] = _focal(bank, v) if k == _FOCAL else _fill(copy.deepcopy(sk[k]), v, REQUIRED_ONE.get(program, set()), k)
    return state


def cf_questions(bank: Bank, program: str, v: str) -> list[tuple[str, QSpec, Any]]:
    """All of `program`'s questions for a content-free state (see the module doc for template expansion)."""
    items = []
    for spec in bank.program(program).values():
        ph = spec.placeholders()["fields"]
        fields = {}
        for f in ph:
            if f == "industry":
                continue
            fields[f] = {"when": v, "kind": v, "headline": v} if f == "prior_event" else v
        if spec.suffix == "k" and program in ("P1", "M"):
            for code, text in bank.ff12.items():
                name, q = spec.expand(index=code, fields={**fields, "industry": text})
                items.append((name, spec, q))
            continue
        index = 0 if spec.suffix is not None else None
        opts = dict(zip(spec.placeholders()["options"], REID_LABELS)) or None
        name, q = spec.expand(index=index, fields=fields or None, option_labels=opts)
        items.append((name, spec, q))
    return items


def cf_requests(bank: Bank, t: pd.Timestamp, programs: tuple[str, ...] | None = None) -> list[Request]:
    """One request per (program, value) -- per (D question, value) for D -- labelled as program 'CF'."""
    applies = [p.strip() for p in bank.state_schemas["CF"]["applies_to"].split(",")]
    values = bank.conventions["content_free"]["values"]
    out = []
    for prog in (programs or applies):
        for vi, v in enumerate(values):
            if prog == "D":
                for qname, (base, sk) in D_STATES.items():
                    spec = bank.q(f"D.{qname}")
                    state = cf_state(bank, base, v) if base is not None else _fill(copy.deepcopy(sk), v, set())
                    opts = dict(zip(spec.placeholders()["options"], REID_LABELS)) or None
                    name, q = spec.expand(option_labels=opts)
                    out.append(Request.build("CF", f"CF:D.{qname}:{vi}", MARKET, t, state, [(name, spec, q)],
                                             tag="CF", relabel="CF"))
                continue
            out.append(Request.build("CF", f"CF:{prog}:{vi}", MARKET, t, cf_state(bank, prog, v),
                                     cf_questions(bank, prog, v), tag="CF", relabel="CF"))
    return out


class ContentFreeProgram:
    """VC-3 as a runner program: one unit asks every content-free request (the states do not depend on the week,
    so running it for several weeks repeats identical, cached requests)."""

    name = "CF"
    version = "CF-v1"
    taint = "D"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, programs: tuple[str, ...] | None = None):
        self.bank = bank or load_bank()
        self.programs = programs

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        t = pd.Timestamp(ctx.t)
        asked = await asyncio.gather(*(ask(s1, r) for r in cf_requests(self.bank, t, self.programs)))
        return [ProgramOutput("CF", MARKET, t, {}, a.trace, [a.request], a.evidence) for a in asked]
