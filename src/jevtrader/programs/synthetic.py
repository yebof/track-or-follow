"""A small synthetic market for offline dry runs of Layer A (with `MockBackend`) and for tests.

Random-walk bars for a few stocks, SPY and two sector ETFs, plus a `WeekContext` factory. Nothing here
resembles real data; it exists so that the runner, programs and point-in-time property tests can run
without the data archive or a paid backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jevtrader.data.blinding import Blinder
from jevtrader.data.news import decision_cutoff
from jevtrader.features.technical import PanelPrices
from jevtrader.programs.base import FirmInfo, WeekContext

INDUSTRIES = ("BusEq", "Money", "Hlth", "Shops", "Enrgy", "Manuf")


@dataclass
class SyntheticWorld:
    bars: pd.DataFrame
    prices: PanelPrices
    symbols: tuple[str, ...]
    firms: dict[str, FirmInfo]
    blinder: Blinder
    sessions: pd.DatetimeIndex

    def decision_dates(self, n_weeks: int, skip_weeks: int = 60) -> list[pd.Timestamp]:
        """Last session of each ISO week, after enough history for 12-1 momentum."""
        s = pd.Series(self.sessions)
        iso = s.dt.isocalendar()
        last = list(s.groupby([iso.year.values, iso.week.values]).max())
        return last[skip_weeks: skip_weeks + n_weeks]

    def context_factory(self, universe: tuple[str, ...] | None = None) -> Callable[[pd.Timestamp], WeekContext]:
        uni = tuple(universe or self.symbols)

        def build(t: pd.Timestamp) -> WeekContext:
            t = pd.Timestamp(t)
            return WeekContext(t=t, cutoff_utc=decision_cutoff(t), universe=uni, prices=self.prices,
                               raw_prices=self.prices, blinder=self.blinder, firms=self.firms)

        return build

    def with_bars(self, bars: pd.DataFrame) -> SyntheticWorld:
        return SyntheticWorld(bars, PanelPrices(bars), self.symbols, self.firms, self.blinder, self.sessions)


def synthetic_world(n_symbols: int = 12, n_sessions: int = 420, seed: int = 0, start: str = "2023-01-02") -> SyntheticWorld:
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=n_sessions)
    symbols = tuple(f"S{i:02d}" for i in range(n_symbols))
    etfs = ("SPY", "XLK", "XLF")
    rows = []
    for k, sym in enumerate(symbols + etfs):
        drift = rng.normal(0.0003, 0.0004)
        vol = 0.01 if sym in etfs else rng.uniform(0.012, 0.03)
        r = rng.normal(drift, vol, n_sessions)
        close = 50.0 * (1 + k % 7) * np.exp(np.cumsum(r))
        volume = rng.lognormal(14, 0.4, n_sessions)
        for d, c, v, e in zip(sessions, close, volume, r):
            rows.append((sym, d, c * (1 - abs(e) / 2), c * (1 + abs(e)), c * (1 - abs(e)), c, v))
    bars = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    firms = {s: FirmInfo(industry=INDUSTRIES[i % len(INDUSTRIES)], size_rank=i * 7 + 1,
                         sector_etf="XLK" if i % 2 == 0 else "XLF") for i, s in enumerate(symbols)}
    blinder = Blinder({s: [s] for s in symbols + etfs})
    return SyntheticWorld(bars, PanelPrices(bars), symbols, firms, blinder, sessions)
