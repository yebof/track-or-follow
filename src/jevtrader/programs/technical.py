"""Technical analyst T (design_v1 §5.3): one request per stock-week, trusted buckets only.

The reference program of the framework. State: `{note, focal_company, price_action, market}` where
`price_action` is `features.technical.describe` (ratios and buckets, no price levels) on closes up to the
decision date, and `market` holds the SPY one-month return with its 5-year percentile bucket plus the top
M.regime option text read from the M-work blackboard (a T-tainted entry, so R1 holds).

Its S1 features (x_trend = sum v(l) p(l) with the bank value map, the four clipped Noul logits, and the
forecast-type `technical_view` for V-outlook) are answer-level phi maps, so Layer B computes them from
answers.parquet (one definition, learn.features); T emits no aggregates. Code technical factors are Layer B's.
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from jevtrader.features.technical import describe, technical_snapshot
from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Evidence, ProgramOutput, Request, WeekContext, ask
from jevtrader.s1.types import canonical_json

log = logging.getLogger("jevtrader.programs")
MARKET_MONTH = 21  # sessions
MARKET_HISTORY = 1260  # 5 years of sessions for the percentile bucket


class TechnicalProgram:
    name = "T"
    version = "T-v1"
    taint = "T"
    depends_on = ("M",)  # the market regime on the blackboard ("not available" without M: the runner refuses that
    week_ordered = False  # unless GeneratorSpec.allow_missing_deps, and the MANIFEST records it)
    memory_reads = ()

    def __init__(self, bank: Bank | None = None, market_symbol: str = "SPY"):
        self.bank = bank or load_bank()
        self.market_symbol = market_symbol
        self.specs = self.bank.program("T")

    # ------------------------------------------------------------------ state parts
    def market_field(self, ctx: WeekContext) -> tuple[dict[str, str], list[Evidence]]:
        close = ctx.prices.close[self.market_symbol].loc[: pd.Timestamp(ctx.t)].dropna()
        ret = close.iloc[-1] / close.iloc[-1 - MARKET_MONTH] - 1 if len(close) > MARKET_MONTH else None
        hist = (close / close.shift(MARKET_MONTH) - 1).dropna().iloc[-MARKET_HISTORY:]
        broad = f"{render.fmt_pct(ret)} ({render.hist_bucket(render.trailing_percentile(hist, ret))} compared with the past 5 years)"
        ev = [Evidence(f"px:{self.market_symbol}", ctx.cutoff_utc, text=broad)]
        regime = render.NOT_AVAILABLE
        entry = ctx.memory.work("M.regime", max_taint="T") if ctx.memory is not None else None
        if entry is not None and entry.value.get("probs"):
            probs = entry.value["probs"]
            regime = self.bank.option_text("M.regime", max(probs, key=probs.get))
            ev.append(Evidence("bb:M.regime", entry.available_at, text=regime))
        else:  # no M in the run, or M abstained on the regime: every T state of the week says so
            log.warning("T %s: no M.regime answer on the blackboard; market regime is %r", pd.Timestamp(ctx.t).date(),
                        render.NOT_AVAILABLE)
        return {"broad US market over the past month": broad, "market regime": regime}, ev

    # ------------------------------------------------------------------ states
    def states(self, ctx: WeekContext, names: list[str] | None = None) -> dict[str, tuple[dict, list[Evidence]]]:
        """symbol -> (T state, evidence) for `names` (default: the run's stock names). Percentile buckets are
        cross-sectional over the full U_t, so a subset run renders exactly a full run's states. Names without a
        close on the decision date or a quarter of history get no state (Layer B: abstain)."""
        uni = list(ctx.universe)
        sector = {s: ctx.firms[s].sector_etf for s in uni if s in ctx.firms and ctx.firms[s].sector_etf}
        snap = technical_snapshot(ctx.prices, ctx.t, uni, sector_etf=sector or None, market=self.market_symbol)
        market, market_ev = self.market_field(ctx)
        out = {}
        for sym in (names if names is not None else ctx.stock_names()):
            if sym not in snap.index:
                continue
            firm = ctx.firms[sym]
            price_action = describe(snap.loc[sym])
            state = render.envelope(
                self.bank, "T",
                focal_company=render.focal_company(self.bank, ctx.blinder.code_name(sym), firm.industry, firm.size_rank),
                price_action=price_action,
                market=market,
            )
            out[sym] = (state, [Evidence(f"px:{sym}", ctx.cutoff_utc, text=canonical_json(price_action)), *market_ev])
        return out

    # ------------------------------------------------------------------ run
    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        items = [(q.name, q, q.to_question()) for q in self.specs.values()]

        async def one(sym: str, state: dict, ev: list[Evidence]) -> ProgramOutput:
            asked = await ask(s1, Request.build("T", "T", sym, ctx.t, state, items, ev))
            return ProgramOutput("T", sym, pd.Timestamp(ctx.t), {}, asked.trace, [asked.request], asked.evidence)

        return list(await asyncio.gather(*(one(s, st, ev) for s, (st, ev) in self.states(ctx).items())))
