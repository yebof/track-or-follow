"""Look-ahead probes D (design_v1 §5.7, §13.7; design_v2_delta LA-2): diagnostics only, never inputs to decisions.

**D1 recall and LAP** (`RecallProbeProgram`, "D"). Unblinded, evidence-free states that name the company and the
holding period of decision t: start = e_t (the session after d_t), end = e_{t+1} (the session after the next
decision), as ISO dates, i.e. exactly the period of the T1 label.
- stage D1: `D.recall_beats_median` on `{company_name, period}`;
- stage LAP: `D.recall_direction` {beat_median, trailed_median, cannot_know} on the same state, in its own request
  (so the emulator's answer to the Noul cannot condition it); LAP = 1 - P(cannot_know), U - D = P(beat) - P(trailed);
- stage D1-market: `D.recall_market_up` on `{period}` (one MARKET request).
The company name is the SEC registrant name valid at d_t (`formerNames` dates), cleaned as the blinder cleans it.
The names asked about are the run's stock names (the S30 subset or the monthly 20-name schedule is chosen by the
run). The window-start rule stays on `recall_beats_median` (pre-registered).

**D2 re-identification** (`ReidProbeProgram`, "D2"). `D.reid_company` asked on the blinded T and F-B states the
programs actually send (same builders, so the same bytes), with the true name and 4 decoys of the same FF12 industry
and adjacent raw-market-value rank in U_t (nearest ranks first; filled from all of U_t if the industry is small),
shuffled by a fixed hash (salt `D2-www27`), plus `cannot_tell`. `reid_year_request` builds `D.reid_year` for a
blinded N1 item state (the news programs own that state builder). Answers are keyed by the real names; the entity
column is the true symbol, so scoring needs no extra table.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from jevtrader.data.blinding import clean_name
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import ProgramOutput, Request, WeekContext, ask
from jevtrader.programs.context import ROOT, decision_dates, previous_decision
from jevtrader.s1.types import Choice

MARKET = "MARKET"
SUB_DIR = ROOT / "data" / "raw" / "sec" / "submissions"
D2_SALT = "D2-www27"
N_DECOYS = 4


# ------------------------------------------------------------------------------------ names and periods
@lru_cache(maxsize=2048)
def _submission(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def company_name(symbol: str, t: pd.Timestamp, sub_dir: Path = SUB_DIR) -> str:
    """The SEC registrant name valid at t (a former name while it was in use), cleaned; the symbol if unknown."""
    d = _submission(str(Path(sub_dir) / f"{symbol}.json"))
    t = pd.Timestamp(t)
    for fn in d.get("formerNames") or []:
        lo = pd.Timestamp(fn.get("from")).tz_localize(None) if fn.get("from") else None
        hi = pd.Timestamp(fn.get("to")).tz_localize(None) if fn.get("to") else None
        if (lo is None or lo <= t) and (hi is None or t < hi):
            return clean_name(fn["name"])
    return clean_name(d.get("name") or symbol)


def next_decision(sessions: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    """The decision date of the ISO week after t's (its last session); None unless the calendar has a later session,
    i.e. unless that week is complete in the calendar."""
    idx = pd.DatetimeIndex(sessions)
    t = pd.Timestamp(t)
    week = tuple(t.isocalendar())[:2]
    later = [d for d in decision_dates(idx[idx > t]) if tuple(d.isocalendar())[:2] != week]
    if not later or not (idx > later[0]).any():
        return None
    return later[0]


def probe_period(sessions: pd.DatetimeIndex, t: pd.Timestamp) -> dict[str, str] | None:
    """{start: e_t, end: e_{t+1}} as ISO dates (the T1 label's holding period); None past the calendar's end."""
    idx = pd.DatetimeIndex(sessions)
    t = pd.Timestamp(t)
    e_t = idx[idx > t]
    nd = next_decision(idx, t)
    if not len(e_t) or nd is None:
        return None
    e_next = idx[idx > nd]
    if not len(e_next):
        return None
    return {"start": e_t[0].date().isoformat(), "end": e_next[0].date().isoformat()}


# ------------------------------------------------------------------------------------ D1 + LAP
class RecallProbeProgram:
    name = "D"
    version = "D-v1"
    taint = "D"  # diagnostic: never written to memory, never a feature
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()
    STAGES = ("D1", "LAP", "D1-market")

    def __init__(self, bank: Bank | None = None, stages: tuple[str, ...] = STAGES, sub_dir: Path = SUB_DIR):
        self.bank = bank or load_bank()
        unknown = set(stages) - set(self.STAGES)
        if unknown:
            raise ValueError(f"unknown probe stages {sorted(unknown)}")
        self.stages, self.sub_dir = stages, sub_dir

    def _item(self, qname: str):
        spec = self.bank.q(f"D.{qname}")
        return [(spec.name, spec, spec.to_question())]

    def requests(self, ctx: WeekContext) -> list[Request]:
        t = pd.Timestamp(ctx.t)
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        period = probe_period(sessions, t)
        if period is None:
            return []
        out = []
        for sym in ctx.stock_names():
            state = {"company_name": company_name(sym, t, self.sub_dir), "period": dict(period)}
            if "D1" in self.stages:
                out.append(Request.build("D", "D1", sym, t, state, self._item("recall_beats_median"), tag="D"))
            if "LAP" in self.stages:
                out.append(Request.build("D", "LAP", sym, t, state, self._item("recall_direction"), tag="D"))
        if "D1-market" in self.stages:
            out.append(Request.build("D", "D1-market", MARKET, t, {"period": dict(period)},
                                     self._item("recall_market_up"), tag="D"))
        return out

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        asked = await asyncio.gather(*(ask(s1, r) for r in self.requests(ctx)))
        t = pd.Timestamp(ctx.t)
        return [ProgramOutput("D", a.request.symbol, t, {}, a.trace, [a.request], a.evidence) for a in asked]


# ------------------------------------------------------------------------------------ D2 re-identification
def decoys(ctx: WeekContext, symbol: str, n: int = N_DECOYS) -> list[str]:
    """`n` U_t members of the same FF12 industry nearest in raw-market-value rank (then symbol), filled from all of
    U_t by rank distance when the industry has fewer."""
    me = ctx.firms[symbol]
    others = [s for s in ctx.universe if s != symbol and s in ctx.firms]
    key = lambda s: (abs(ctx.firms[s].size_rank - me.size_rank), s)
    same = sorted((s for s in others if ctx.firms[s].industry == me.industry), key=key)
    rest = sorted((s for s in others if s not in set(same)), key=key)
    return (same + rest)[:n]


def reid_question(bank: Bank, true_name: str, decoy_names: list[str], seed: str) -> Choice:
    """`D.reid_company` with its placeholders filled and the five names shuffled by sha256(seed | name)."""
    spec = bank.q("D.reid_company")
    names = [true_name, *decoy_names]
    labels = []
    for nm in names:  # distinct labels even if two registrants clean to the same name
        lab = nm
        k = 2
        while lab in labels:
            lab, k = f"{nm} ({k})", k + 1
        labels.append(lab)
    ph = spec.placeholders()["options"]  # "<true name>", "<decoy 1>", ... in bank order
    _, q = spec.expand(option_labels=dict(zip(ph, labels)))
    order = sorted(labels, key=lambda lab: hashlib.sha256(f"{seed}|{lab}".encode()).hexdigest())
    rest = [k for k in q.options if k not in set(labels)]  # cannot_tell stays last
    return Choice(q.instructions, {k: q.options[k] for k in [*order, *rest]})


def reid_year_request(bank: Bank, t: pd.Timestamp, entity: str, state: dict, stage: str = "D2:N1") -> Request:
    """`D.reid_year` on a blinded N1 item state (built by the news programs)."""
    spec = bank.q("D.reid_year")
    return Request.build("D2", stage, entity, t, state, [(spec.name, spec, spec.to_question())], tag="D2")


class ReidProbeProgram:
    name = "D2"
    version = "D2-v1"
    taint = "D"
    depends_on = ("M",)  # T states carry M's market regime; the probe must see the bytes T sends
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, sources: tuple[str, ...] = ("T", "FB"), sub_dir: Path = SUB_DIR):
        self.bank = bank or load_bank()
        self.sources, self.sub_dir = sources, sub_dir

    def requests(self, ctx: WeekContext) -> list[Request]:
        from jevtrader.programs.fundamentals import FundamentalsProgram
        from jevtrader.programs.technical import TechnicalProgram

        t = pd.Timestamp(ctx.t)
        states = {}
        if "T" in self.sources:
            states["T"] = TechnicalProgram(self.bank).states(ctx)
        if "FB" in self.sources:
            states["FB"] = FundamentalsProgram(self.bank).states(ctx)
        spec = self.bank.q("D.reid_company")
        out = []
        for sym in ctx.stock_names():
            names = [company_name(s, t, self.sub_dir) for s in [sym, *decoys(ctx, sym)]]
            q = reid_question(self.bank, names[0], names[1:], f"{D2_SALT}|{t.date()}|{sym}")
            for src, st in states.items():
                if sym in st:
                    state, ev = st[sym]
                    out.append(Request.build("D2", f"D2:{src}", sym, t, state, [(spec.name, spec, q)], ev, tag="D2"))
        return out

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        asked = await asyncio.gather(*(ask(s1, r) for r in self.requests(ctx)))
        t = pd.Timestamp(ctx.t)
        return [ProgramOutput("D2", a.request.symbol, t, {}, a.trace, [a.request], a.evidence) for a in asked]


__all__ = [
    "RecallProbeProgram",
    "ReidProbeProgram",
    "company_name",
    "decoys",
    "next_decision",
    "previous_decision",
    "probe_period",
    "reid_question",
    "reid_year_request",
]
