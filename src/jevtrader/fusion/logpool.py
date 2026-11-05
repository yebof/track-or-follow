"""Learned logarithmic opinion pool with exact additive attribution.

    logit P(y=1 | x) = b_r + sum_k w_k * x_k,   x_k = logit of source k's calibrated probability,

where r indexes a regime (per-regime intercept) and w_k >= 0 (a source can be down-weighted to zero
but not inverted; inversion would make attributions contradict the source's own judgment). Weights are
fit by weighted, L2-regularized logistic regression on past (x, y) pairs only. The objective is the weighted
*sum* of losses plus the penalty (divided by sum(w) for conditioning), so decay weights never silently weaken
the prior (design v1, validity m13). Kept for the two-level pool variant V-2L (design §6.6). Because the pooled
logit is linear in the inputs, `contributions` decomposes each decision exactly:
    logit P = b_r + sum_k c_k,   c_k = w_k * x_k.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from jevtrader.fusion.calibration import EPS, sigmoid


class LogPool:
    def __init__(self, sources: list[str], n_regimes: int = 1, l2: float = 5.0, nonneg: bool = True, prior_w: float = 1.0):
        self.sources = list(sources)
        self.n_regimes = n_regimes
        self.l2, self.nonneg = l2, nonneg
        self.w = np.full(len(sources), prior_w / max(1, len(sources)))
        self.b = np.zeros(n_regimes)
        self._prior_w = self.w.copy()

    def fit(self, X: np.ndarray, y: np.ndarray, regime: np.ndarray | None = None, sample_weight: np.ndarray | None = None) -> "LogPool":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, k = X.shape
        if n < 50 or len(np.unique(y)) < 2:
            return self
        r = np.zeros(n, dtype=int) if regime is None else np.asarray(regime, dtype=int)
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        R = np.eye(self.n_regimes)[r]  # one-hot regimes

        def f(theta):
            w, b = theta[:k], theta[k:]
            z = X @ w + R @ b
            q = np.clip(sigmoid(z), EPS, 1 - EPS)
            loss = -(sw * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / sw.sum()
            tw = sw.sum()
            reg = self.l2 * (((w - self._prior_w) ** 2).sum() + (b**2).sum()) / tw
            g = sw * (q - y) / tw
            gw = X.T @ g + 2 * self.l2 * (w - self._prior_w) / tw
            gb = R.T @ g + 2 * self.l2 * b / tw
            return loss + reg, np.concatenate([gw, gb])

        bounds = [(0.0, None) if self.nonneg else (None, None)] * k + [(None, None)] * self.n_regimes
        res = minimize(f, x0=np.concatenate([self.w, self.b]), jac=True, method="L-BFGS-B", bounds=bounds)
        self.w, self.b = res.x[:k].copy(), res.x[k:].copy()
        return self

    def logit(self, X: np.ndarray, regime: np.ndarray | int = 0) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        r = np.broadcast_to(np.asarray(regime, dtype=int), (X.shape[0],))
        return X @ self.w + self.b[r]

    def predict(self, X: np.ndarray, regime: np.ndarray | int = 0) -> np.ndarray:
        return sigmoid(self.logit(X, regime))

    def contributions(self, x: np.ndarray, regime: int = 0) -> dict[str, float]:
        """Exact decomposition of one decision's pooled logit: {'intercept': b_r, source: w_k * x_k}."""
        x = np.asarray(x, dtype=float)
        out = {"intercept": float(self.b[regime])}
        out.update({s: float(wk * xk) for s, wk, xk in zip(self.sources, self.w, x)})
        return out
