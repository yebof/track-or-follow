"""Classical, non-LLM baselines on the same universe, clock, costs and engine as the system (design_v1 section 13.3).

Each baseline maps (decision date d, technical snapshot of U_d at d) -> long-only target weights. The snapshot covers
the point-in-time universe U_d as known at c_t: there is NO filter on next-session tradability (V-M2). A name that
cannot be filled at e_t stays in cash inside the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(snap: pd.DataFrame, **_) -> dict[str, float]:
    n = len(snap)
    return {s: 1.0 / n for s in snap.index} if n else {}


def top_k_by(snap: pd.DataFrame, col: str, k: int = 20, ascending: bool = False, **_) -> dict[str, float]:
    s = snap[col].dropna().sort_values(ascending=ascending, kind="mergesort")
    top = s.index[:k]
    return {x: 1.0 / len(top) for x in top} if len(top) else {}


def momentum_12_1(snap: pd.DataFrame, k: int = 20, **_) -> dict[str, float]:
    return top_k_by(snap, "ret_12_1", k)


def short_term_reversal(snap: pd.DataFrame, k: int = 20, **_) -> dict[str, float]:
    return top_k_by(snap, "ret_1w", k, ascending=True)


def low_volatility(snap: pd.DataFrame, k: int = 20, **_) -> dict[str, float]:
    return top_k_by(snap, "vol_60d", k, ascending=True)


def random_k(members: list[str], k: int, rng: np.random.Generator) -> dict[str, float]:
    """B7 Random-20: k names drawn uniformly from U_t, equal-weighted (one weekly redraw)."""
    members = sorted(members)
    pick = rng.choice(len(members), size=min(k, len(members)), replace=False)
    return {members[i]: 1.0 / len(pick) for i in pick}


def top_k_scores(scores: pd.Series, k: int = 20) -> dict[str, float]:
    """Equal weight on the k highest scores; ties broken by symbol for determinism."""
    s = scores.dropna()
    s = s.iloc[np.lexsort((s.index.to_numpy(), -s.to_numpy()))]
    top = s.index[:k]
    return {x: 1.0 / len(top) for x in top} if len(top) else {}


# ------------------------------------------------------------------------------------------------ PE-5
VM_WINDOW = 22  # sessions of realized variance (moreira2017volatilitymanaged: previous month, 22 daily returns, p.6)
VM_MIN_HISTORY = 252  # sessions of history before c is estimated; weight 1 until then
VM_CAP = 1.0  # no leverage: the paper's "no leverage" variant caps weights at 1 (Table IV-V, p.15-16)


def vol_managed_weight(returns: pd.Series, d: pd.Timestamp, window: int = VM_WINDOW,
                       min_history: int = VM_MIN_HISTORY, cap: float = VM_CAP) -> float:
    """Exposure of a volatility-managed book at the close of d (design_v2_delta PE-5): w = min(cap, c / RV_d).

    RV_d is the mean squared daily return of the underlying over the `window` sessions ending at d. The paper sets c
    so that managed and unmanaged books have the same full-sample volatility (p.6, fn. 6), which is not point in
    time; here c_d is the mean squared daily return over ALL sessions up to d (the long-run variance known at d), so
    w = 1 when recent variance is at or below its long-run level. Only returns dated <= d are read."""
    r = returns[returns.index <= pd.Timestamp(d)].dropna()
    if len(r) < max(window, min_history):
        return cap
    rv = float((r.iloc[-window:] ** 2).mean())
    c = float((r ** 2).mean())
    return cap if rv <= 0 else float(min(cap, c / rv))


def ew_member_returns(close: pd.DataFrame, members: list[str], d: pd.Timestamp) -> pd.Series:
    """Daily returns (sessions <= d) of an equal-weight portfolio of `members` rebalanced daily: the underlying of the
    vol-managed EW-U100 book, from adjusted closes up to d only."""
    cols = [m for m in members if m in close.columns]
    px = close.loc[: pd.Timestamp(d), cols]
    r = px.pct_change(fill_method=None)
    return r.mean(axis=1, skipna=True).iloc[1:]


def vol_managed(book: dict[str, float], weight: float) -> dict[str, float]:
    """Scale a long-only book by the exposure weight (<= 1); the rest is cash at DFF in the engine."""
    if not 0.0 <= weight <= 1.0 + 1e-12:
        raise ValueError(f"vol-managed exposure must be in [0, 1], got {weight}")
    return {s: x * weight for s, x in book.items()}


BASELINES = {
    "equal_weight": equal_weight,
    "momentum_12_1": momentum_12_1,
    "reversal_1w": short_term_reversal,
    "low_vol": low_volatility,
}
