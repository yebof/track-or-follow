"""Raw (as-traded) and adjusted daily price panels, split events and symbol lives (design_v1 section 2.4, V-B1, V-m2).

Two price bases, never mixed:
- **adjusted** bars (Alpaca `adjustment=all`: splits, dividends, spin-offs) for returns and ratio features only;
- **raw** bars (`adjustment=raw`) for every level: market value, yields, dollar volume, price buckets. Adjusted
  history is rescaled by every later corporate action, so a level computed from it at d depends on events after d.

Split events are recovered from the two downloads. At a split with ratio k on session s, Alpaca multiplies the
adjusted volume of every earlier session by k and divides earlier adjusted prices by k, while dividends and spin-offs
leave the volume untouched. So g_s = V(s-1)/V(s) with V = adjusted volume / raw volume equals k exactly on split days
and 1 otherwise; the price ratio f_s = F(s-1)/F(s), F = raw close / adjusted close, must agree. Both ratios only
involve sessions s-1 and s (later adjustments cancel), so a split is known from its ex-date on. The rule fails when
a split coincides with a spin-off or merger (Alpaca then adjusts volume by a combined or wrong factor); those events
are set from the issuer's 8-K in SPLIT_OVERRIDES, and a test checks every event against XBRL share counts.

A symbol's life ends at its first gap of more than MAX_GAP sessions (V-m2): Alpaca reuses tickers, and a later
issuer's bars under the same ticker (APC, FB, NFX) must not be forward-filled into the old company's life. The rule
is causal: trimming the bars up to d gives the same result as trimming all bars and then cutting at d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from jevtrader.features.technical import PanelPrices

ROOT = Path(__file__).resolve().parents[3]
RAW_BARS = ("data/raw/bars/all_daily_raw_2018_2026.parquet",)  # stocks and ETFs
ADJ_BARS = ("data/raw/bars/sp500_daily_2018_2026.parquet", "data/raw/bars/etf_daily_2018_2026.parquet")
MAX_GAP = 10  # sessions
CALENDAR_SYMBOL = "SPY"

# Split ratios seen in practice (forward and reverse); the volume ratio is exact, so a tight tolerance separates
# them from spin-offs that happen to be near 1.25 or 1.5 (DOV 2018: 1.238, DD 2019: 1.487).
_FWD = sorted({Fraction(k) for k in range(2, 101)} | {Fraction(p, q) for p, q in
               [(3, 2), (5, 4), (4, 3), (5, 3), (5, 2), (7, 4), (7, 5), (6, 5), (8, 5), (9, 5)]})
SPLIT_RATIOS = np.array(sorted([float(x) for x in _FWD] + [1.0 / k for k in range(2, 201)]
                               + [float(1 / x) for x in _FWD if x.denominator > 1]))

# Events the volume/price-ratio rule gets wrong, set from the issuer's 8-K (ratio = new shares per old share; None =
# no split). Each action was announced before its ex-date, so using it from the ex-date on is point in time.
# tests/test_prices.py cross-checks every detected and rejected event of the universe against XBRL cover counts.
SPLIT_OVERRIDES: dict[tuple[str, str], float | None] = {
    # UTX completed the Carrier/Otis distributions and the Raytheon merger (2.3348 UTX shares per RTN share) and
    # became RTX; no reverse split (8-K 0001140361-20-008397). Alpaca's adjustment has f = g = 0.5; the cover count
    # rose 1.76x.
    ("RTX", "2020-04-03"): None,
    # "the one-for-three reverse stock split ... previously approved by the Company's stockholders and the Board on
    # May 23, 2019", effective 2019-06-01 right after the Corteva distribution (8-K 0001193125-19-163322). The
    # combined adjustment (0.4725) is not a split ratio, so the rule rejects it.
    ("DD", "2019-06-03"): 1 / 3,
    # "the previously announced reverse stock split ... every two shares ... combined into one share", effective
    # 2026-06-29 with the Aerospace spin-off (8-K 0000773840-26-000084). Volume ratio 0.5, price ratio 0.954.
    ("HON", "2026-06-29"): 0.5,
}


def load_bars(kind: str, root: Path = ROOT) -> pd.DataFrame:
    """Long bars (symbol, date as Timestamp, open, high, low, close, volume, ...). kind: 'raw' or 'adjusted'."""
    files = {"raw": RAW_BARS, "adjusted": ADJ_BARS}[kind]
    b = pd.concat([pd.read_parquet(Path(root) / f) for f in files], ignore_index=True)
    b["date"] = pd.to_datetime(b["date"])
    b["symbol"] = b["symbol"].astype(str)
    return b.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)


def sessions_from(bars: pd.DataFrame, symbol: str = CALENDAR_SYMBOL) -> pd.DatetimeIndex:
    """Trading calendar = the sessions on which the calendar symbol (SPY) traded."""
    s = bars.loc[bars.symbol == symbol, "date"]
    if s.empty:
        s = bars["date"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(s).unique()))


def life_ends(bars: pd.DataFrame, sessions: pd.DatetimeIndex, max_gap: int = MAX_GAP) -> pd.Series:
    """symbol -> last bar before its first gap of more than `max_gap` missing sessions (NaT if no such gap)."""
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    out = {}
    for sym, d in bars.groupby("symbol")["date"]:
        p = pos.reindex(d.values).to_numpy()
        p = p[np.isfinite(p)]
        missing = np.diff(p) - 1
        big = np.flatnonzero(missing > max_gap)
        out[sym] = sessions[int(p[big[0]])] if len(big) else pd.NaT
    return pd.Series(out, dtype="datetime64[ns]")


def trim_at_first_gap(bars: pd.DataFrame, sessions: pd.DatetimeIndex, max_gap: int = MAX_GAP) -> pd.DataFrame:
    """Drop every bar after a symbol's first gap of more than `max_gap` sessions (ticker reuse, V-m2)."""
    ends = life_ends(bars, sessions, max_gap).dropna()
    if ends.empty:
        return bars
    cut = bars["symbol"].map(ends)
    keep = cut.isna() | (bars["date"] <= cut)
    return bars[keep.to_numpy()].reset_index(drop=True)


def cut_at(bars: pd.DataFrame, ends: pd.Series) -> pd.DataFrame:
    """Drop bars after each symbol's life end (`ends`: symbol -> last date; missing/NaT = no cut). Used to give the
    adjusted bars exactly the cut computed on the raw bars."""
    ends = pd.to_datetime(ends.dropna())
    if ends.empty:
        return bars.reset_index(drop=True)
    cut = bars["symbol"].map(ends)
    return bars[(cut.isna() | (bars["date"] <= cut)).to_numpy()].reset_index(drop=True)


def gap_violations(bars: pd.DataFrame, sessions: pd.DatetimeIndex, max_gap: int = MAX_GAP) -> dict[str, int]:
    """symbol -> largest number of missing sessions between consecutive bars, for symbols above `max_gap`."""
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    out = {}
    for sym, d in bars.groupby("symbol")["date"]:
        p = pos.reindex(d.values).to_numpy()
        p = p[np.isfinite(p)]
        if len(p) > 1 and (np.diff(p) - 1).max() > max_gap:
            out[sym] = int((np.diff(p) - 1).max())
    return out


def split_candidates(raw: pd.DataFrame, adj: pd.DataFrame, tol: float = 1e-3, price_tol: float = 0.05) -> pd.DataFrame:
    """Every session whose adjusted volume factor changes (|log g| > 0.01): symbol, date, g, f, the nearest split
    ratio and whether the rule accepts it (`ok`). Used by `split_events` and by the XBRL cross-check test."""
    cols = ["symbol", "date", "close", "volume"]
    m = raw[cols].merge(adj[cols], on=["symbol", "date"], suffixes=("_r", "_a")).sort_values(["symbol", "date"])
    m = m[(m.volume_r > 0) & (m.volume_a > 0) & (m.close_r > 0) & (m.close_a > 0)]
    F = m.close_r / m.close_a
    V = m.volume_a / m.volume_r
    g = V.groupby(m.symbol).shift(1) / V
    f = F.groupby(m.symbol).shift(1) / F
    cand = m.assign(g=g, f=f)[np.abs(np.log(g)) > 0.01].copy()
    if cand.empty:
        return pd.DataFrame(columns=["symbol", "date", "g", "f", "ratio", "ok"])
    idx = np.abs(np.log(cand.g.to_numpy()[:, None] / SPLIT_RATIOS[None, :])).argmin(axis=1)
    cand["ratio"] = SPLIT_RATIOS[idx]
    cand["ok"] = (np.abs(np.log(cand.g / cand.ratio)) < tol) & (np.abs(np.log(cand.f / cand.ratio)) < price_tol)
    return cand[["symbol", "date", "g", "f", "ratio", "ok"]].reset_index(drop=True)


def split_events(raw: pd.DataFrame, adj: pd.DataFrame, tol: float = 1e-3, price_tol: float = 0.05,
                 overrides: dict | None = None) -> pd.DataFrame:
    """Split events (symbol, date = first session on the new basis, ratio = new shares per old share): the sessions
    where g is a split ratio within `tol` and f agrees within `price_tol`, corrected by SPLIT_OVERRIDES for
    symbols present in `raw`."""
    overrides = SPLIT_OVERRIDES if overrides is None else overrides
    cand = split_candidates(raw, adj, tol, price_tol)
    ev = cand.loc[cand.ok.astype(bool), ["symbol", "date", "ratio"]]
    have = set(raw.symbol)
    fixed = {(s, pd.Timestamp(d)): r for (s, d), r in overrides.items() if s in have}
    ev = ev.loc[np.array([(s, pd.Timestamp(d)) not in fixed for s, d in zip(ev.symbol, ev.date)], dtype=bool)]
    add = pd.DataFrame([(s, d, r) for (s, d), r in fixed.items() if r is not None],
                       columns=["symbol", "date", "ratio"])
    out = pd.concat([ev, add], ignore_index=True) if len(add) else ev
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


@dataclass
class SplitHistory:
    """Cumulative split factors: how many shares one share held just after `after` became by `upto`."""

    events: pd.DataFrame  # symbol, date, ratio

    def __post_init__(self):
        self._by = {s: (pd.DatetimeIndex(g.date), g.ratio.to_numpy(float)) for s, g in self.events.groupby("symbol")}

    def factor(self, symbol: str, after: pd.Timestamp, upto: pd.Timestamp) -> float:
        """Product of split ratios with ex-date in (after, upto]. Uses only events up to `upto`."""
        ev = self._by.get(symbol)
        if ev is None or pd.isna(after):
            return 1.0
        d, r = ev
        m = (d > pd.Timestamp(after)) & (d <= pd.Timestamp(upto))
        return float(np.prod(r[m])) if m.any() else 1.0


@dataclass
class PricePanels:
    """Everything price-related a decision at d may read (only rows <= d are ever used)."""

    adjusted: PanelPrices
    raw: PanelPrices
    splits: SplitHistory
    sessions: pd.DatetimeIndex
    life_end: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))

    def raw_close_asof(self, symbol: str, d: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
        """Last raw close on or before d (and its session), inside the symbol's life."""
        if symbol not in self.raw.close:
            return None
        c = self.raw.close[symbol].loc[: pd.Timestamp(d)].dropna()
        return (float(c.iloc[-1]), c.index[-1]) if len(c) else None


def build_panels(raw: pd.DataFrame, adj: pd.DataFrame, max_gap: int = MAX_GAP) -> PricePanels:
    """Trim both bases at each symbol's first long gap (the same cut for both) and pivot them into panels."""
    sessions = sessions_from(raw)
    ends = life_ends(raw, sessions, max_gap)
    raw_t = cut_at(raw, ends)
    adj_t = cut_at(adj, ends)
    return PricePanels(adjusted=PanelPrices(adj_t), raw=PanelPrices(raw_t), splits=SplitHistory(split_events(raw, adj)),
                       sessions=sessions, life_end=ends.dropna())


def load_price_panels(root: Path = ROOT) -> PricePanels:
    return build_panels(load_bars("raw", root), load_bars("adjusted", root))
