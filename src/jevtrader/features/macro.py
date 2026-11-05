"""Point-in-time macro, market-regime and policy-uncertainty features (design_v1 sections 2.4 and 5.6; V-m1, F19).

Availability rules (conservative, to avoid look-ahead):
- Revised macro series (CPI, payrolls, ...): ALFRED vintages; value for observation period p is the one whose
  vintage `realtime_start` <= decision date d (latest such vintage).
- Daily market series from FRED (yields, spreads, VIX): observation date <= d - 1 day (FRED posts most daily series
  the next day). DTWEXBGS is published weekly and DCOILWTICO with a lag, so both use d - 7 days.
- NFCI is not used: it is revised every week and FRED serves only its current vintage.
- EPU daily index: date <= d - 2 days. GPR daily index: date <= d - 7 days (the "recent" file lags several days).
  Both are read from the incremental archives (`*_daily_archive.csv`, scripts/download_sec_macro.py indices).
- FOMC: only meetings of the pre-announced tentative schedule whose press release predates d
  (data/raw/macro/fomc_calendar.csv); no jobs-report flag.

Every indicator is rendered as its percentile within a FIXED trailing 5-year window ending at its latest available
observation (history from 2005, so the window is always full), plus its direction over the past month. The window
never grows with the sample, so the reference distribution means the same thing in 2020 and 2026.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VINTAGE_SERIES = {
    "CPIAUCSL": "CPI (all items)",
    "CPILFESL": "core CPI",
    "UNRATE": "unemployment rate",
    "PAYEMS": "nonfarm payrolls",
    "INDPRO": "industrial production",
    "RSAFS": "retail sales",
    "GDPC1": "real GDP",
    "PCEPILFE": "core PCE price index",
    "ICSA": "initial jobless claims",
}
# How each revised series is measured before taking percentiles (question bank, M state): growth series use the
# year-over-year change, payrolls the 3-month average monthly change, claims their 4-week average level.
VINTAGE_TRANSFORM = {"CPIAUCSL": "yoy", "CPILFESL": "yoy", "PCEPILFE": "yoy", "INDPRO": "yoy", "RSAFS": "yoy",
                     "GDPC1": "yoy", "PAYEMS": "avg3_change", "UNRATE": "level", "ICSA": "avg4_level"}
DAILY_SERIES = {
    "DGS10": "10-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "DGS3MO": "3-month T-bill yield",
    "T10Y2Y": "10y-2y yield curve slope",
    "DFF": "fed funds effective rate",
    "VIXCLS": "VIX",
    "BAA10Y": "Baa corporate bond spread over 10-year Treasury",
    "AAA10Y": "Aaa corporate bond spread over 10-year Treasury",
    "DTWEXBGS": "broad trade-weighted dollar",
    "DCOILWTICO": "WTI crude oil",
    "T10YIE": "10-year breakeven inflation",
}
DAILY_LAG_DAYS = {"DTWEXBGS": 7, "DCOILWTICO": 7}  # default 1 (V-m1)
EPU_LAG_DAYS, GPR_LAG_DAYS = 2, 7
WINDOW_YEARS = 5
MIN_COVERAGE = 0.9  # share of the 5-year window that must hold data, else the percentile is NaN
DIRECTION_Z = 0.5  # |1-month change| below 0.5 sd of 1-month changes in the window -> "little changed"


def macro_store_sha256(root: Path) -> str:
    """SHA-256 over (relative path, file sha256) of every file under `root`/data/raw/macro, links followed: the macro
    inputs a run read (for run manifests and the live refresh timeline). The archive's value must not change when a
    live refresh runs (`live.refresh` writes revised FRED vintages to the live root only)."""
    import hashlib

    base = Path(root) / "data" / "raw" / "macro"
    h = hashlib.sha256()
    for f in sorted(x for x in base.rglob("*") if x.is_file()):
        h.update(f.relative_to(base).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def _periods_per_year(idx: pd.DatetimeIndex) -> int:
    step = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float)) if len(idx) > 2 else 30
    return 52 if step < 10 else 12 if step < 45 else 4


def transform(s: pd.Series, how: str) -> pd.Series:
    """Apply a VINTAGE_TRANSFORM to one vintage's observation series (indexed by observation period)."""
    s = s.sort_index()
    ppy = _periods_per_year(s.index)
    if how == "yoy":
        return (s / s.shift(ppy) - 1.0).dropna()
    if how == "avg3_change":
        return ((s - s.shift(3)) / 3.0).dropna()
    if how == "avg4_level":
        return s.rolling(4).mean().dropna()
    return s


def pctile_fixed_window(s: pd.Series, years: int = WINDOW_YEARS) -> float:
    """Percentile of the last value among the values in (last date - `years`, last date]; NaN if the window is
    less than MIN_COVERAGE covered by data."""
    s = s.dropna()
    if s.empty:
        return np.nan
    end = s.index[-1]
    lo = end - pd.DateOffset(years=years)
    w = s[s.index > lo]
    if s.index[0] > lo + pd.DateOffset(days=int(365.25 * years * (1 - MIN_COVERAGE))):
        return np.nan
    return float((w <= w.iloc[-1]).mean())


def direction(change: float, sd: float) -> str:
    if not np.isfinite(change) or not np.isfinite(sd) or sd <= 0:
        return "unknown"
    z = change / sd
    return "rising" if z > DIRECTION_Z else "falling" if z < -DIRECTION_Z else "little changed"


def _one_month_changes(s: pd.Series, years: int = WINDOW_YEARS) -> pd.Series:
    """Changes over one month (by date) within the fixed window, for the direction threshold."""
    w = s[s.index > s.index[-1] - pd.DateOffset(years=years)]
    prev = s.reindex(w.index - pd.DateOffset(days=30), method="ffill")
    return pd.Series(w.values - prev.values, index=w.index).dropna()


def _read_index_archive(root: Path, name: str) -> pd.Series:
    arc = root / f"{name}_daily_archive.csv"
    if arc.exists():
        a = pd.read_csv(arc, parse_dates=["date"])
        return pd.Series(a.value.values, index=a.date).sort_index()
    if name == "epu":  # first download, before the incremental archive existed
        e = pd.read_csv(root / "epu_daily.csv")
        return pd.Series(e.daily_policy_index.values,
                         index=pd.to_datetime(dict(year=e.year, month=e.month, day=e.day))).sort_index()
    g = pd.read_excel(root / "gpr_daily_recent.xls")
    return pd.Series(g.GPRD.values, index=pd.to_datetime(g.date)).sort_index()


def load_fomc_calendar(path: Path) -> pd.DataFrame:
    """Pre-announced FOMC meetings only (scheduled == True), with the announcement date."""
    c = pd.read_csv(path, parse_dates=["meeting_start", "decision_date", "announced"])
    return c[c.scheduled.astype(str).str.lower() == "true"].reset_index(drop=True)


class MacroStore:
    def __init__(self, root: Path):
        root = Path(root)
        self.vintages: dict[str, pd.DataFrame] = {}
        self.daily: dict[str, pd.Series] = {}
        for s in VINTAGE_SERIES:
            obs = json.loads((root / "fred" / f"{s}.json").read_text())["observations"]
            df = pd.DataFrame(obs)
            df = df[df.value != "."]
            df["value"] = df.value.astype(float)
            for c in ("realtime_start", "date"):
                df[c] = pd.to_datetime(df[c])
            self.vintages[s] = df[["date", "realtime_start", "value"]].sort_values("realtime_start")
        for s in DAILY_SERIES:
            obs = json.loads((root / "fred" / f"{s}.json").read_text())["observations"]
            df = pd.DataFrame(obs)
            df = df[df.value != "."]
            ser = pd.Series(df.value.astype(float).values, index=pd.to_datetime(df.date)).sort_index()
            self.daily[s] = ser[~ser.index.duplicated(keep="last")]
        self.epu = _read_index_archive(root, "epu")
        self.gpr = _read_index_archive(root, "gpr")
        cal = root / "fomc_calendar.csv"
        self.fomc = load_fomc_calendar(cal) if cal.exists() else None

    # ------------------------------------------------------------------ point-in-time accessors
    def vintage_series(self, s: str, d: pd.Timestamp) -> pd.Series:
        """Observation-period -> value as known at the close of d."""
        v = self.vintages[s]
        v = v[v.realtime_start <= pd.Timestamp(d)]
        v = v.groupby("date").tail(1)  # sorted by realtime_start: the latest vintage per period
        return v.set_index("date").value.sort_index()

    def daily_asof(self, s: str, d: pd.Timestamp, lag_days: int | None = None) -> pd.Series:
        lag = DAILY_LAG_DAYS.get(s, 1) if lag_days is None else lag_days
        return self.daily[s].loc[: pd.Timestamp(d) - pd.Timedelta(days=lag)]

    def epu_asof(self, d: pd.Timestamp) -> pd.Series:
        return self.epu.loc[: pd.Timestamp(d) - pd.Timedelta(days=EPU_LAG_DAYS)]

    def gpr_asof(self, d: pd.Timestamp) -> pd.Series:
        return self.gpr.loc[: pd.Timestamp(d) - pd.Timedelta(days=GPR_LAG_DAYS)]

    def fed_decision_within(self, d: pd.Timestamp, days: int = 7) -> bool | None:
        """A pre-announced FOMC decision in (d, d + days], using only schedules announced before d."""
        if self.fomc is None:
            return None
        d = pd.Timestamp(d).normalize()
        c = self.fomc[self.fomc.announced < d]
        return bool(((c.decision_date > d) & (c.decision_date <= d + pd.Timedelta(days=days))).any())

    def cash_rate_daily(self, sessions: pd.DatetimeIndex) -> pd.Series:
        """Daily cash return DFF/252 for each session, from the latest DFF observation strictly before the session
        (the rate in force overnight into it; FRED posts DFF the next morning)."""
        dff = self.daily["DFF"]
        prev = dff.reindex(pd.DatetimeIndex(sessions) - pd.Timedelta(days=1), method="ffill")
        return pd.Series(prev.values / 100.0 / 252.0, index=pd.DatetimeIndex(sessions)).fillna(0.0)

    # ------------------------------------------------------------------------------ snapshot
    def _vintage_record(self, s: str, d: pd.Timestamp) -> dict | None:
        how = VINTAGE_TRANSFORM[s]
        x = transform(self.vintage_series(s, d), how)
        if len(x) < 2:
            return None
        prev = transform(self.vintage_series(s, d - pd.Timedelta(days=30)), how)
        change = float(x.iloc[-1] - prev.iloc[-1]) if len(prev) else np.nan
        w = x[x.index > x.index[-1] - pd.DateOffset(years=WINDOW_YEARS)]
        sd = float(w.diff().dropna().std()) if len(w) > 3 else np.nan
        return {"latest_period": x.index[-1].date().isoformat(), "measure": how, "value": float(x.iloc[-1]),
                "pctile_5y": pctile_fixed_window(x), "change_1m": change, "direction_1m": direction(change, sd)}

    def _daily_record(self, v: pd.Series) -> dict | None:
        v = v.dropna()
        if len(v) < 30:
            return None
        one_m = v.loc[: v.index[-1] - pd.Timedelta(days=30)]
        change = float(v.iloc[-1] - one_m.iloc[-1]) if len(one_m) else np.nan
        ch = _one_month_changes(v)
        return {"latest_date": v.index[-1].date().isoformat(), "value": float(v.iloc[-1]),
                "pctile_5y": pctile_fixed_window(v), "change_1m": change,
                "direction_1m": direction(change, float(ch.std()) if len(ch) > 3 else np.nan)}

    def snapshot(self, d: pd.Timestamp) -> dict:
        """Indicator name -> {value, pctile_5y, change_1m, direction_1m, ...} as known at d. Levels and dates are
        for audit; states render only the percentile bucket and the direction."""
        d = pd.Timestamp(d)
        out: dict = {}
        for s, name in VINTAGE_SERIES.items():
            rec = self._vintage_record(s, d)
            if rec:
                out[name] = rec
        for s, name in DAILY_SERIES.items():
            rec = self._daily_record(self.daily_asof(s, d))
            if rec:
                out[name] = rec
        for label, ser in (("economic policy uncertainty (EPU, 30d avg)", self.epu_asof(d)),
                           ("geopolitical risk (GPR, 30d avg)", self.gpr_asof(d))):
            m30 = ser.rolling("30D").mean()
            rec = self._daily_record(m30)
            if rec:
                out[label] = rec
        out["calendar"] = {"fed_decision_within_week": self.fed_decision_within(d)}
        return out


def market_trend(spy_close: pd.Series, d: pd.Timestamp) -> dict:
    """S&P 500 trend for the M state from adjusted SPY closes up to d: 1-month return, its fixed-window percentile
    among 1-month returns (NaN until 5 years of bars exist) and direction."""
    c = spy_close.loc[: pd.Timestamp(d)].dropna()
    if len(c) < 22:
        return {}
    r21 = (c / c.shift(21) - 1.0).dropna()
    return {"value": float(r21.iloc[-1]), "pctile_5y": pctile_fixed_window(r21),
            "direction_1m": direction(float(r21.iloc[-1]), float(r21.std()))}
