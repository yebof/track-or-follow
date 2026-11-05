"""Targets with label maturity (design §2.3) and the ex-ante idiosyncratic vol they use.

Clock (§2.1): decision t is on session d_t with cutoff c_t = 16:00 ET; the book is filled at the open of e_t (the
next session) and held until the open of e_{t+1}. A label is therefore realized only at open(e_{t+1}), and that
instant (UTC) is its `available_at`. `LabelStore.mature(c)` releases a row only if `available_at < c`, which makes a
label of decision t usable from decision t+2 on. Maturity is decided on timestamps, never on week indices, so holiday
weeks need no special case. Every learner and every outcome-dependent statistic reads labels through `mature`
(tests/test_label_maturity.py).

Targets:
  T1      y = 1[r > m_t],  r = O(e_{t+1}) / O(e_t) - 1 on adjusted opens,  m_t = median of r over U_t
  T3      z = 1[(r - m_t) / sigma < -2.5],  sigma = ex-ante weekly idiosyncratic vol at d_t (`idio_vol_weekly`)
  T1^(4)  1[4-week open-to-open return > its median over U_t], available at open(e_{t+4})
  AR      abnormal return of stock-day news over sessions d0..d0+1 (`abnormal_returns`; evaluation only)
Delistings follow the engine (V-m14): a name whose life ends inside the holding period exits at its last close times
(1 + delisting return); the return must be explicit unless the exit is a known M&A exit (0%), otherwise
`DelistingError` is raised so a silent 0% can never hide a bankruptcy in the labels either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jevtrader.backtest.engine import DelistingError, PriceBook, weekly_decision_dates

ET = "America/New_York"
LABEL_COLUMNS = [
    "t", "symbol", "entry", "exit", "r", "m", "y", "sigma", "u", "z", "available_at",
    "r4", "m4", "y4", "available_at4", "delisted",
]


def open_time_utc(session) -> pd.Timestamp:
    """09:30 America/New_York on `session`, in UTC (DST-safe: localized at the wall-clock time itself)."""
    return pd.Timestamp(f"{pd.Timestamp(session).date()} 09:30").tz_localize(ET).tz_convert("UTC")


def close_time_utc(session) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(session).date()} 16:00").tz_localize(ET).tz_convert("UTC")


def _as_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("cutoffs must be tz-aware (UTC); naive timestamps are ambiguous")
    return ts.tz_convert("UTC")


# ------------------------------------------------------------------------------------ market model
def rolling_beta(ret: pd.DataFrame, market: str = "SPY", window: int = 252, min_periods: int = 60) -> pd.DataFrame:
    """Trailing OLS beta of every column on `market`, using only rows up to and including each date (causal).
    The market series is masked to each stock's own observed days so both moments use the same sample."""
    if market not in ret:
        raise KeyError(f"market column {market!r} missing")
    m = ret[market].to_numpy()[:, None]
    mk = pd.DataFrame(np.where(ret.notna(), m, np.nan), index=ret.index, columns=ret.columns)
    roll = {"window": window, "min_periods": min_periods}
    mx, my = ret.rolling(**roll).mean(), mk.rolling(**roll).mean()
    cov = (ret * mk).rolling(**roll).mean() - mx * my
    var = (mk * mk).rolling(**roll).mean() - my * my
    return cov / var.where(var > 0)


def idio_vol_weekly(
    close: pd.DataFrame, market: str = "SPY", com: float = 60.0, beta_window: int = 252,
    min_periods: int = 60, ewm_min_periods: int = 20,
) -> pd.DataFrame:
    """Ex-ante weekly idiosyncratic vol sigma_{i,d} (sessions x symbols), point-in-time at each session's close.

    Residuals of daily log returns on the market with a trailing beta, an EWMA of squared residuals with a 60-day
    center of mass, then daily vol x sqrt(5) (calibrated_learning §2, as adopted in design §2.3). Every operation is
    causal, so the value at d uses closes up to d only (tests/test_trailing_only.py)."""
    lr = np.log(close.astype(float)).diff()
    beta = rolling_beta(lr, market, beta_window, min_periods)
    resid = lr - beta.mul(lr[market], axis=0)
    ev = (resid**2).ewm(com=com, min_periods=ewm_min_periods).mean()
    return np.sqrt(ev * 5.0)


# ------------------------------------------------------------------------------------ label store
@dataclass
class LabelStore:
    """Realized labels, one row per (decision t, symbol in U_t) with a defined entry price.

    `mature(cutoff)` is the only accessor learners may use; `at(t)` is for after-the-fact evaluation (ICs, log-loss)
    and must never feed a decision."""

    table: pd.DataFrame

    def __post_init__(self):
        missing = set(LABEL_COLUMNS) - set(self.table.columns)
        if missing:
            raise ValueError(f"label table misses {sorted(missing)}")
        tb = self.table.copy()
        tb["t"] = pd.to_datetime(tb["t"])
        for c in ("available_at", "available_at4"):
            tb[c] = pd.to_datetime(tb[c], utc=True)
        self.table = tb.sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)

    @classmethod
    def build(
        cls,
        prices: PriceBook,
        decisions: Sequence[pd.Timestamp],
        universe: Mapping[pd.Timestamp, Sequence[str]] | Callable[[pd.Timestamp], Sequence[str]],
        sigma: pd.DataFrame | None = None,
        tail_k: float = 2.5,
        delisting_returns: Mapping[str, float] | None = None,
        ma_exits: frozenset[str] | set[str] = frozenset(),
    ) -> LabelStore:
        """`decisions` must be weekly decision dates (last session of an ISO week) on `prices.sessions`; the next
        decision date, hence e_{t+1}, is read from the same weekly calendar."""
        dr = {**{s: 0.0 for s in ma_exits}, **dict(delisting_returns or {})}
        uni = universe if callable(universe) else (lambda t: universe[pd.Timestamp(t)])
        cal = weekly_decision_dates(prices.sessions, prices.sessions[0], prices.sessions[-1])
        pos = {d: k for k, d in enumerate(cal)}
        frames = []
        for t in sorted(pd.Timestamp(x) for x in decisions):
            if t not in pos:
                raise ValueError(f"{t.date()} is not a weekly decision date of the price calendar")
            k = pos[t]
            entry = prices.next_session(t)
            exit1 = prices.next_session(cal[k + 1]) if k + 1 < len(cal) else None
            if entry is None or exit1 is None:
                continue  # label not realized inside the data
            exit4 = prices.next_session(cal[k + 4]) if k + 4 < len(cal) else None
            syms = list(dict.fromkeys(uni(t)))
            o0 = prices.open.reindex(columns=syms).loc[entry].to_numpy(dtype=float)
            r, delisted = cls._period_return(prices, syms, o0, exit1, dr)
            m = np.nanmedian(r) if np.isfinite(r).any() else np.nan
            y = np.where(np.isfinite(r), (r > m).astype(float), np.nan)
            sig = (
                sigma.reindex(index=[t], columns=syms).iloc[0].to_numpy(dtype=float)
                if sigma is not None else np.full(len(syms), np.nan)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (r - m) / np.where(sig > 0, sig, np.nan)
            z = np.where(np.isfinite(u), (u < -tail_k).astype(float), np.nan)
            if exit4 is not None:
                r4, _ = cls._period_return(prices, syms, o0, exit4, dr)
                m4 = np.nanmedian(r4) if np.isfinite(r4).any() else np.nan
                y4 = np.where(np.isfinite(r4), (r4 > m4).astype(float), np.nan)
                av4 = open_time_utc(exit4)
            else:
                r4, m4, y4, av4 = np.full(len(syms), np.nan), np.nan, np.full(len(syms), np.nan), pd.NaT
            f = pd.DataFrame({
                "t": t, "symbol": syms, "entry": entry, "exit": exit1, "r": r, "m": m, "y": y, "sigma": sig,
                "u": u, "z": z, "available_at": open_time_utc(exit1), "r4": r4, "m4": m4, "y4": y4,
                "available_at4": av4, "delisted": delisted,
            })
            frames.append(f[np.isfinite(o0)])  # a name without an entry open was never buyable: no label
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLUMNS)
        return cls(table)

    @staticmethod
    def _period_return(prices: PriceBook, syms: list[str], o0: np.ndarray, exit_session, dr: dict):
        o1 = prices.open.reindex(columns=syms).loc[exit_session].to_numpy(dtype=float, copy=True)
        delisted = np.isfinite(o0) & ~np.isfinite(o1)
        for j in np.flatnonzero(delisted):
            s = syms[j]
            if s not in dr:
                raise DelistingError(f"{s} delisted during a label period without an explicit delisting return or "
                                     "M&A exit (design_v1 section 9.2, V-m14)")
            o1[j] = prices.close.at[prices.last[s], s] * (1.0 + dr[s])
        with np.errstate(divide="ignore", invalid="ignore"):
            return o1 / o0 - 1.0, delisted

    # ------------------------------------------------------------------ accessors
    def mature(self, cutoff_utc, horizon: int = 1) -> pd.DataFrame:
        """Rows whose label for `horizon` (1 or 4 weeks) was realized strictly before `cutoff_utc`."""
        c = _as_utc(cutoff_utc)
        tb = self.table
        if horizon == 1:
            ok = (tb["available_at"] < c) & tb["y"].notna()
        elif horizon == 4:
            ok = (tb["available_at4"] < c) & tb["y4"].notna()
        else:
            raise ValueError("horizon must be 1 or 4")
        return tb[ok.to_numpy()].reset_index(drop=True)

    def at(self, t) -> pd.DataFrame:
        """All realized labels of decision t, regardless of maturity (evaluation only)."""
        return self.table[self.table["t"] == pd.Timestamp(t)].reset_index(drop=True)

    def restrict(self, start, end) -> LabelStore:
        """Labels of decisions in [start, end] only: protocol P-clean learns only from its own window."""
        t = self.table["t"]
        return LabelStore(self.table[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))])


# ------------------------------------------------------------------------------ abnormal returns (C2)
def _event_session(created_at: pd.Timestamp, sessions: pd.DatetimeIndex) -> int | None:
    """Index of d0: the session of `created_at` if before 16:00 ET that day, otherwise the next session."""
    local = pd.Timestamp(created_at).tz_convert(ET)
    day = pd.Timestamp(local.date())
    side = "left" if local.hour < 16 else "right"
    k = sessions.searchsorted(day, side=side)
    return int(k) if k < len(sessions) else None


def abnormal_returns(
    prices: PriceBook, events: pd.DataFrame, market: str = "SPY", window: tuple[int, int] = (0, 1),
    beta_window: int = 252, min_periods: int = 60,
) -> pd.DataFrame:
    """AR over sessions d0+window[0] .. d0+window[1] of (r_i - beta_i r_market), simple daily close-to-close returns
    (design §13.1 C2). beta_i is the trailing beta ending the session before the window starts, so it is point-in-time
    for the placebo window (-5, -1) too. `events` needs `symbol` and tz-aware `created_at`. Evaluation only: AR is
    never a trading label. Rows that cannot be computed get NaN."""
    ret = prices.close.pct_change(fill_method=None)
    beta = rolling_beta(ret, market, beta_window, min_periods)
    rv, bv, mv = ret.to_numpy(), beta.to_numpy(), ret[market].to_numpy()
    col = {s: j for j, s in enumerate(ret.columns)}
    sess = prices.sessions
    a, b = window
    d0s, ars, avail = [], [], []
    for sym, ts in zip(events["symbol"], pd.to_datetime(events["created_at"], utc=True)):
        k0 = _event_session(ts, sess)
        lo, hi = (None, None) if k0 is None else (k0 + a, k0 + b)
        j = col.get(sym)
        if k0 is None or j is None or min(lo, k0) < 1 or hi >= len(sess):
            d0s.append(sess[k0] if k0 is not None else pd.NaT)
            ars.append(np.nan)
            avail.append(pd.NaT)
            continue
        be = bv[min(lo, k0) - 1, j]
        seg = rv[lo : hi + 1, j] - be * mv[lo : hi + 1]
        d0s.append(sess[k0])
        ars.append(float(seg.sum()) if np.isfinite(seg).all() else np.nan)
        avail.append(close_time_utc(sess[hi]))
    out = events.copy()
    out["d0"], out["ar"], out["available_at"] = d0s, ars, pd.to_datetime(pd.Series(avail, index=out.index), utc=True)
    return out
