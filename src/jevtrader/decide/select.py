"""Selection with the band guard (design §3.4, §9.1).

    1. Rank U_t by the combined logit L = L^T + capped L^W (R), and separately by the trusted logit L^T (R^T).
       Ties are broken by a fixed hash of the symbol, which no text can influence.
    2. Forced in: R^T <= K - J and not vetoed.
    3. Eligible: R^T <= K + J.
    4. Fill the remaining slots with the eligible names that have the best combined L.

With incumbents (D-sized), a held name also stays if R <= K + buffer and R^T <= K + J. Forced names are placed
first, then staying incumbents by combined rank, then the best eligible names, at most K in total. Placing forced
names first is what keeps Proposition §3.4 (ii) true when incumbents alone could fill the book.

Proposition §3.4 (checked in tests/test_band_guard_bound.py): (i) a selected name has R^T <= K + J; (ii) a name with
R^T <= K - J is selected unless vetoed. Both hold for any values of the untrusted inputs because R^T is computed from
trusted inputs only and the rules above are applied after ranking.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

TIEBREAK_SALT = "tiebreak-www27"


@lru_cache(maxsize=65536)
def tiebreak_key(symbol: str, salt: str = TIEBREAK_SALT) -> str:
    return hashlib.sha256((salt + symbol).encode()).hexdigest()


def rank_desc(values: pd.Series, salt: str = TIEBREAK_SALT) -> pd.Series:
    """1 = largest value; exact ties are ordered by the symbol hash (never by input order). NaN ranks last."""
    v = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    keys = np.array([tiebreak_key(s, salt) for s in values.index])
    order = np.lexsort((keys, np.where(np.isnan(v), np.inf, -v)))
    ranks = np.empty(len(v), dtype=int)
    ranks[order] = np.arange(1, len(v) + 1)
    return pd.Series(ranks, index=values.index)


@dataclass(frozen=True)
class Selection:
    names: tuple[str, ...]  # selected names in the order they were placed (forced, incumbents, fill)
    forced: tuple[str, ...]
    kept_incumbents: tuple[str, ...]
    vetoed: tuple[str, ...]
    band: pd.Series  # symbol -> "forced" | "eligible" | "outside"
    rank_T: pd.Series
    rank_L: pd.Series
    trusted_topK: tuple[str, ...]  # trusted top-K by R^T (the D-sized normalizer set, text-independent)
    K: int
    J: float


def select(
    scores, K: int = 20, J: float = 10, incumbents: Iterable[str] = (), vetoed: Iterable[str] = (),
    incumbent_buffer: int | None = None,
) -> Selection:
    """`scores.frame` must hold `L_T` and `L` per symbol of U_t (or precomputed `R_T` / `R`). `J` may be math.inf
    (unguarded); J = 0 gives the trusted top-K. `incumbent_buffer` enables the D-sized incumbent rule."""
    fr = scores.frame if hasattr(scores, "frame") else scores
    rT = fr["R_T"].astype(int) if "R_T" in fr else rank_desc(fr["L_T"])
    rL = fr["R"].astype(int) if "R" in fr else rank_desc(fr["L"])
    veto = set(vetoed) & set(fr.index)
    lo, hi = K - J, K + J
    band = pd.Series(np.where(rT <= lo, "forced", np.where(rT <= hi, "eligible", "outside")), index=fr.index)
    by_L = list(rL.sort_values(kind="mergesort").index)
    rTd, rLd = rT.to_dict(), rL.to_dict()  # plain lookups (the selection is replayed many times by the bound checks)
    forced = [s for s in by_L if rTd[s] <= lo and s not in veto]
    chosen = list(forced)
    taken = set(chosen)
    kept: list[str] = []
    if incumbent_buffer is not None:
        inc = set(incumbents) & set(fr.index)
        for s in by_L:
            if len(chosen) >= K:
                break
            if s in inc and s not in veto and s not in taken and rLd[s] <= K + incumbent_buffer and rTd[s] <= hi:
                chosen.append(s)
                taken.add(s)
                kept.append(s)
    for s in by_L:
        if len(chosen) >= K:
            break
        if s not in taken and s not in veto and rTd[s] <= hi:
            chosen.append(s)
            taken.add(s)
    topk = tuple(rT.sort_values(kind="mergesort").index[:K])
    return Selection(
        names=tuple(chosen), forced=tuple(forced), kept_incumbents=tuple(kept), vetoed=tuple(sorted(veto)),
        band=band, rank_T=rT, rank_L=rL, trusted_topK=topk, K=K, J=J if math.isfinite(J) else math.inf,
    )
