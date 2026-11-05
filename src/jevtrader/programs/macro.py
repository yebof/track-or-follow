"""Macro analyst M (design_v1 §5.6): one request per week, trusted buckets only.

State `{note (market note), indicators, calendar}`:
- `indicators`: every series of `features.macro.MacroStore.snapshot` (NFCI is already dropped there; revised series
  as known at d_t from ALFRED vintages, daily FRED series with their lags, EPU and GPR 30-day averages) plus the S&P
  500 trend, each rendered '<percentile bucket> compared with the past 5 years; <rising|falling|little changed> over
  the past month'. Growth series are measured as the bank says (year-over-year change; payrolls as the 3-month
  average monthly change; claims as the 4-week average), and the measure is named next to the series. No dates, no
  levels: the snapshot's values and dates are used only for evidence rows.
- `calendar`: whether a pre-announced FOMC decision falls within the next week (schedules announced before d_t).

All 18 questions are asked in bank order (regime, growth, inflation, fed_direction, stress, market_view, then one
`industry_tailwind_k` per FF12 industry). Layer B maps the tailwind answers to the per-stock feature x_mac from the
MARKET rows (`learn.features.S1_TRUSTED_MAP`), so M emits no aggregates. Its descriptive answers go to the M-work
blackboard (T reads `M.regime`, a T-tainted entry, so rule R1 holds; the overlay variant reads `M.stress`).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from jevtrader.features.macro import DAILY_LAG_DAYS, DAILY_SERIES, VINTAGE_SERIES, VINTAGE_TRANSFORM, market_trend
from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Evidence, ProgramOutput, QSpec, Request, WeekContext, ask

log = logging.getLogger("jevtrader.programs")
MARKET = "MARKET"
SP500_TREND = "S&P 500 trend (1-month return)"
MEASURE = {"yoy": "change over the past year", "avg3_change": "average monthly change over the past 3 months",
           "avg4_level": "4-week average", "level": None}
EPU_NAME, GPR_NAME = "economic policy uncertainty (EPU, 30d avg)", "geopolitical risk (GPR, 30d avg)"
BLACKBOARD_QUESTIONS = ("regime", "growth", "inflation", "fed_direction", "stress")


def indicator_label(name: str) -> str:
    """The state key of a snapshot indicator: the series name, plus its measure for revised series."""
    for sid, n in VINTAGE_SERIES.items():
        if n == name and MEASURE.get(VINTAGE_TRANSFORM[sid]):
            return f"{n}, {MEASURE[VINTAGE_TRANSFORM[sid]]}"
    return name


# every indicator key the M state can hold, in state order (content-free probe skeletons use it)
INDICATOR_KEYS: tuple[str, ...] = (*(indicator_label(n) for n in VINTAGE_SERIES.values()), *DAILY_SERIES.values(),
                                   EPU_NAME, GPR_NAME, SP500_TREND)
CALENDAR_KEY = "Fed rate decision within the next week"


def render_indicator(pctile: float | None, direction: str | None) -> str:
    """'<bucket> compared with the past 5 years; <direction> over the past month' (bank §B, M)."""
    ok = pctile is not None and np.isfinite(pctile)
    level = f"{render.hist_bucket(pctile)} compared with the past 5 years" if ok else \
        "no 5-year comparison available"
    move = f"{direction} over the past month" if direction and direction != "unknown" else \
        "direction over the past month not available"
    return f"{level}; {move}"


def expand_industries(bank: Bank, spec: QSpec) -> list[tuple[str, QSpec, object]]:
    """One question per FF12 industry, in the bank's industry order (template suffix _k)."""
    out = []
    for code, text in bank.ff12.items():
        name, q = spec.expand(index=code, fields={"industry": text})
        out.append((name, spec, q))
    return out


def program_items(bank: Bank, program: str) -> list[tuple[str, QSpec, object]]:
    """All questions of a market program (M or P1) in bank order, `_k` templates expanded over FF12."""
    items = []
    for spec in bank.program(program).values():
        items += expand_industries(bank, spec) if spec.suffix == "k" else [(spec.name, spec, spec.to_question())]
    return items


class MacroProgram:
    name = "M"
    version = "M-v1"
    taint = "T"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, market_symbol: str = "SPY"):
        self.bank = bank or load_bank()
        self.market_symbol = market_symbol

    # ------------------------------------------------------------------ state
    def state(self, ctx: WeekContext) -> tuple[dict, list[Evidence]]:
        t = pd.Timestamp(ctx.t)
        snap = ctx.macro.snapshot(t)
        indicators: dict[str, str] = {}
        ev: list[Evidence] = []
        vint_ids = {n: s for s, n in VINTAGE_SERIES.items()}
        daily_ids = {n: s for s, n in DAILY_SERIES.items()}
        for name, rec in snap.items():
            if name == "calendar" or not isinstance(rec, dict):
                continue
            text = render_indicator(rec.get("pctile_5y"), rec.get("direction_1m"))
            indicators[indicator_label(name)] = text
            ev.append(Evidence(f"macro:{vint_ids.get(name) or daily_ids.get(name) or name}",
                               self._available_at(ctx, name, rec, vint_ids, daily_ids), text=text))
        trend = market_trend(ctx.prices.close[self.market_symbol], t)
        indicators[SP500_TREND] = render_indicator(trend.get("pctile_5y"), trend.get("direction_1m"))
        ev.append(Evidence(f"px:{self.market_symbol}", ctx.cutoff_utc, text=indicators[SP500_TREND]))
        fed = snap.get("calendar", {}).get("fed_decision_within_week")
        calendar = {CALENDAR_KEY: render.NOT_AVAILABLE if fed is None else ("yes" if fed else "no")}
        if fed is not None and ctx.macro.fomc is not None:
            ann = ctx.macro.fomc.announced
            ann = ann[ann < t.normalize()]
            if len(ann):
                ev.append(Evidence("macro:fomc_schedule", pd.Timestamp(ann.max()).tz_localize("UTC"),
                                   text=calendar[CALENDAR_KEY]))
        state = render.envelope(self.bank, "M", indicators=indicators, calendar=calendar)
        return state, ev

    @staticmethod
    def _available_at(ctx: WeekContext, name: str, rec: dict, vint_ids: dict, daily_ids: dict) -> pd.Timestamp:
        """When the value shown was public: the vintage's realtime_start for revised series; the observation date
        plus its publication lag for daily series and indices (all strictly before the cutoff)."""
        t = pd.Timestamp(ctx.t)
        if name in vint_ids:
            v = ctx.macro.vintages[vint_ids[name]]
            rs = v.loc[v.realtime_start <= t, "realtime_start"]
            day = rs.max() if len(rs) else t
        elif name in daily_ids:
            day = pd.Timestamp(rec["latest_date"]) + pd.Timedelta(days=DAILY_LAG_DAYS.get(daily_ids[name], 1))
        else:
            lag = 2 if name == EPU_NAME else 7
            day = pd.Timestamp(rec["latest_date"]) + pd.Timedelta(days=lag)
        return pd.Timestamp(min(pd.Timestamp(day), t)).tz_localize("UTC")

    # ------------------------------------------------------------------ run
    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        if "macro" in ctx.unavailable:  # stale series would render an outage as an unchanged week: abstain
            log.warning("M %s: macro series do not cover the week; M abstains", pd.Timestamp(ctx.t).date())
            return []
        state, ev = self.state(ctx)
        asked = await ask(s1, Request.build("M", "M", MARKET, ctx.t, state, program_items(self.bank, "M"), ev))
        answers = {r.qid: r.probs for r in asked.trace}
        bb = {f"M.{q}": {"probs": answers.get(q)} for q in BLACKBOARD_QUESTIONS}
        tail = {}
        for code in self.bank.ff12:
            p = answers.get(f"industry_tailwind_{code}")
            tail[code] = None if p is None else float(sum(int(k) * v for k, v in p.items()) - 1.0)  # t_g in [-1, 1]
        bb["M.industry_tailwind"] = tail
        return [ProgramOutput("M", MARKET, pd.Timestamp(ctx.t), {}, asked.trace, [asked.request], asked.evidence, bb)]


def stress_expectation(probs: dict | None) -> float | None:
    """E[M.stress] on levels 0..3 (overlay variant; §9.3)."""
    if not probs:
        return None
    e = sum(int(k) * float(v) for k, v in probs.items())
    return e if math.isfinite(e) else None
