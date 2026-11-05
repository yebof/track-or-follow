"""Point-in-time technical features computed in code (Jev is not a calculator; see docs/jev/pages/
model-jaggedness__jev-1.13). Features at decision date d use bars up to and including d only.

Price bases (design_v1 section 2.4, V-B1): returns and ratios come from ADJUSTED bars; any level (dollar volume,
price buckets) comes from RAW bars, because adjusted history is rescaled by corporate actions after d.

`technical_snapshot` returns raw numbers plus cross-sectional percentile ranks within the decision-date
universe; `describe` turns them into named buckets for the System-One state; `code_features` gives the six
trusted code features of section 5.3 / 6.1; `SectorMap` is the trailing-correlation sector-ETF map (V-m6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"ret_1w": 5, "ret_1m": 21, "ret_3m": 63, "ret_6m": 126}


SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")  # SPDR sectors


class PanelPrices:
    """Wide close / open / volume tables (dates x symbols), stocks and ETFs together. One price basis per panel
    (adjusted or raw; see data/prices.py)."""

    def __init__(self, bars: pd.DataFrame):
        b = bars.copy()
        b["date"] = pd.to_datetime(b["date"])
        self.close = b.pivot(index="date", columns="symbol", values="close").sort_index()
        wide = lambda col: (b.pivot(index="date", columns="symbol", values=col).reindex(self.close.index)  # noqa: E731
                            if col in b else pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns))
        self.open = wide("open")
        self.volume = wide("volume")
        self.high = wide("high")
        self.low = wide("low")

    def upto(self, d: pd.Timestamp, n: int) -> pd.DataFrame:
        c = self.close.loc[: pd.Timestamp(d)]
        return c.iloc[-n:]


def _rsi(c: pd.Series, n: int = 14) -> float:
    d = c.diff().dropna()
    if len(d) < n:
        return np.nan
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return float(100.0 if dn == 0 else 100 - 100 / (1 + up / dn))


def technical_snapshot(
    px: PanelPrices, d: pd.Timestamp, symbols: list[str], sector_etf: dict[str, str] | None = None, market: str = "SPY",
    raw: PanelPrices | None = None,
) -> pd.DataFrame:
    """`px` is the ADJUSTED panel. With `raw` given, level features (median dollar volume over 63 sessions) are
    added from as-traded prices."""
    d = pd.Timestamp(d)
    hist = px.close.loc[:d].iloc[-300:]
    vol = px.volume.loc[:d].iloc[-300:]
    idx = hist.index
    rows = []
    mret = hist[market].pct_change() if market in hist else None

    def at(x: pd.Series, n: int) -> float:
        """Close of `x` as of the session n sessions before d (last trade at or before it; NaN before its first
        bar). Windows are anchored on the calendar, not on the stock's own bars, so a halt does not stretch a
        window and a stock and its sector ETF are compared over the same dates."""
        return float(x.asof(idx[-1 - n])) if len(idx) > n and len(x) and idx[-1 - n] >= x.index[0] else np.nan

    for s in symbols:
        c = hist[s].dropna() if s in hist else pd.Series(dtype=float)
        if len(c) < 64 or c.index[-1] != hist.index[-1]:
            continue  # need a quarter of history and a close on d
        r = {"symbol": s}
        for k, n in WINDOWS.items():
            r[k] = c.iloc[-1] / at(c, n) - 1
        r["ret_12_1"] = at(c, 21) / at(c, 252) - 1
        lr = np.log(c).diff()
        r["vol_20d"] = lr.iloc[-20:].std() * np.sqrt(252)
        r["vol_60d"] = lr.iloc[-60:].std() * np.sqrt(252)
        r["rsi14"] = _rsi(c.iloc[-60:])
        r["dist_50dma"] = c.iloc[-1] / c.iloc[-50:].mean() - 1
        r["dist_200dma"] = c.iloc[-1] / c.iloc[-200:].mean() - 1 if len(c) >= 200 else np.nan
        r["drawdown_52w"] = c.iloc[-1] / c.iloc[-252:].max() - 1
        v = vol[s].dropna()
        r["volume_surge"] = v.iloc[-5:].mean() / v.iloc[-60:].mean() if len(v) >= 60 and v.iloc[-60:].mean() > 0 else np.nan
        if mret is not None:
            a = pd.concat([lr.iloc[-252:], np.log1p(mret).iloc[-252:]], axis=1, sort=True).dropna()
            r["beta_1y"] = a.iloc[:, 0].cov(a.iloc[:, 1]) / a.iloc[:, 1].var() if len(a) > 60 else np.nan
        if sector_etf and s in sector_etf and sector_etf[s] in hist:
            e = hist[sector_etf[s]].dropna()
            r["rel_sector_3m"] = (c.iloc[-1] / at(c, 63)) / (at(e, 0) / at(e, 63)) - 1
        if raw is not None and s in raw.close:
            dv = (raw.close[s] * raw.volume[s]).loc[:d].dropna().iloc[-63:]
            r["dollar_volume_63d"] = float(dv.median()) if len(dv) >= 20 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(index=pd.Index([], name="symbol"))
    for col in ["ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12_1", "vol_60d", "volume_surge", "rel_sector_3m",
                "dollar_volume_63d"]:
        if col in df:
            df[col + "_pct"] = df[col].rank(pct=True)
    return df


# --------------------------------------------------------------------------- trusted code features (§5.3)
CODE_FEATURES = ("mom_12_1", "rev_1w", "rev_1m", "vol_60d", "dist_52w_high", "rel_sector_3m")


def code_features(snap: pd.DataFrame) -> pd.DataFrame:
    """The six trusted code technical features (raw values; rank-Gaussian standardization is Layer B's).
    Reversal features are signed so that a past loser scores high; the learners fit the sign anyway."""
    col = lambda c: snap[c] if c in snap else pd.Series(np.nan, index=snap.index)  # noqa: E731
    return pd.DataFrame({
        "mom_12_1": col("ret_12_1"),
        "rev_1w": -col("ret_1w"),
        "rev_1m": -col("ret_1m"),
        "vol_60d": col("vol_60d"),
        "dist_52w_high": col("drawdown_52w"),
        "rel_sector_3m": col("rel_sector_3m"),
    }, index=snap.index)


class SectorMap:
    """Stock -> SPDR sector ETF with the highest correlation of daily log returns over the trailing `window`
    sessions (V-m6; calibrated_learning section 3). Recomputed monthly: the map used at d is the one computed on the
    first session of d's month from returns up to and including that session, so it never reads beyond d."""

    def __init__(self, px: PanelPrices, etfs: tuple[str, ...] = SECTOR_ETFS, window: int = 252, min_obs: int = 200):
        self.px, self.etfs, self.window, self.min_obs = px, [e for e in etfs if e in px.close], window, min_obs
        self._cache: dict[pd.Timestamp, dict[str, str]] = {}

    def month_start(self, d: pd.Timestamp) -> pd.Timestamp:
        idx = self.px.close.index
        d = pd.Timestamp(d)
        m = idx[(idx <= d) & (idx >= d.replace(day=1))]
        return m[0] if len(m) else idx[idx <= d][-1]

    def asof(self, d: pd.Timestamp, symbols: list[str]) -> dict[str, str]:
        ms = self.month_start(d)
        have = self._cache.setdefault(ms, {})
        todo = [s for s in symbols if s not in have and s in self.px.close]
        if todo:
            have.update(self._compute(ms, todo))
        return {s: have[s] for s in symbols if have.get(s)}

    def _compute(self, t: pd.Timestamp, symbols: list[str]) -> dict[str, str]:
        c = self.px.close.loc[:t].iloc[-(self.window + 1):]
        lr = np.log(c[list(dict.fromkeys(symbols + self.etfs))]).diff().iloc[1:]
        E = lr[self.etfs]
        out = {}
        for s in symbols:
            if s in self.etfs:
                continue
            x = lr[s]
            n = (E.notna() & x.notna().to_numpy()[:, None]).sum()
            rho = E.corrwith(x).where(n >= self.min_obs).dropna()
            out[s] = str(rho.idxmax()) if len(rho) else ""
        return out


# ------------------------------------------------------------------------------ semantic buckets
def _pct_bucket(p: float) -> str:
    if not np.isfinite(p):
        return "unknown"
    if p >= 0.9:
        return "top 10% of the universe"
    if p >= 0.7:
        return "upper 30%"
    if p > 0.3:
        return "middle of the pack"
    if p > 0.1:
        return "lower 30%"
    return "bottom 10% of the universe"


def _rsi_bucket(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    return "overbought" if x >= 70 else "strong" if x >= 55 else "neutral" if x > 45 else "weak" if x > 30 else "oversold"


def _dist_bucket(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x > 0.10:
        return "far above"
    if x > 0.02:
        return "above"
    if x >= -0.02:
        return "near"
    if x >= -0.10:
        return "below"
    return "far below"


def describe(row: pd.Series) -> dict:
    """Named-bucket description (value in parentheses) for one symbol's technical snapshot."""
    f = lambda x: "n/a" if not np.isfinite(x) else f"{x:+.1%}"  # noqa: E731
    out = {
        "past week return": f"{f(row.ret_1w)} ({_pct_bucket(row.get('ret_1w_pct', np.nan))})",
        "past month return": f"{f(row.ret_1m)} ({_pct_bucket(row.get('ret_1m_pct', np.nan))})",
        "past 3-month return": f"{f(row.ret_3m)} ({_pct_bucket(row.get('ret_3m_pct', np.nan))})",
        "12-1 month momentum": f"{f(row.ret_12_1)} ({_pct_bucket(row.get('ret_12_1_pct', np.nan))})",
        "RSI(14)": f"{row.rsi14:.0f} ({_rsi_bucket(row.rsi14)})" if np.isfinite(row.rsi14) else "n/a",
        "price vs 50-day average": f"{_dist_bucket(row.dist_50dma)} ({f(row.dist_50dma)})",
        "price vs 200-day average": f"{_dist_bucket(row.dist_200dma)} ({f(row.dist_200dma)})",
        "drawdown from 52-week high": f(row.drawdown_52w),
        "volatility (60d, annualized)": f"{row.vol_60d:.0%} ({_pct_bucket(row.get('vol_60d_pct', np.nan))} by volatility)",
        "recent volume vs normal": ("n/a" if not np.isfinite(row.volume_surge) else
                                    f"{row.volume_surge:.1f}x ({'unusually high' if row.volume_surge > 1.5 else 'unusually low' if row.volume_surge < 0.7 else 'normal'})"),
    }
    if "rel_sector_3m" in row and np.isfinite(row.rel_sector_3m):
        out["3-month performance vs sector"] = f"{f(row.rel_sector_3m)} ({'outperforming' if row.rel_sector_3m > 0.02 else 'underperforming' if row.rel_sector_3m < -0.02 else 'in line with'} its sector)"
    return out
