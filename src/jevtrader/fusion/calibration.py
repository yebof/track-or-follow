"""Outcome calibration: map a (semantically calibrated) judgment probability to P(outcome).

All learners are fit only on (prediction, outcome) pairs whose outcomes were realized before the
decision date, so their predictions are out-of-sample by construction. Work in logit space.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def decay_weights(ages: np.ndarray, half_life: float | None) -> np.ndarray:
    """Exponential down-weighting of older samples; `ages` in the same unit as `half_life` (e.g. weeks)."""
    if not half_life:
        return np.ones_like(ages, dtype=float)
    return 0.5 ** (np.asarray(ages, dtype=float) / half_life)


class PlattCalibrator:
    """P(y=1) = sigmoid(a * logit(p) + b), fit by weighted, L2-regularized log loss (shrinks to a=1, b=0).

    Objective: sum_i w_i loss_i + l2 * ((a-1)^2 + b^2), divided by sum(w) for conditioning. The penalty is scaled by
    sum(w), not by n, so decayed weights do not silently weaken the prior (design v1, validity m13)."""

    def __init__(self, l2: float = 1.0):
        self.l2, self.a, self.b = l2, 1.0, 0.0

    def fit(self, p, y, w=None) -> "PlattCalibrator":
        z, y = logit(p), np.asarray(y, dtype=float)
        w = np.ones_like(z) if w is None else np.asarray(w, dtype=float)
        if len(z) < 10 or len(np.unique(y)) < 2:
            return self

        def f(theta):
            a, b = theta
            q = sigmoid(a * z + b)
            q = np.clip(q, EPS, 1 - EPS)
            loss = -(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / w.sum()
            sw = w.sum()
            reg = self.l2 * ((a - 1.0) ** 2 + b**2) / sw
            g_common = w * (q - y) / sw
            ga = (g_common * z).sum() + 2 * self.l2 * (a - 1.0) / sw
            gb = g_common.sum() + 2 * self.l2 * b / sw
            return loss + reg, np.array([ga, gb])

        r = minimize(f, x0=np.array([self.a, self.b]), jac=True, method="L-BFGS-B")
        self.a, self.b = map(float, r.x)
        return self

    def predict(self, p):
        return sigmoid(self.a * logit(p) + self.b)


class IsotonicCalibrator:
    def __init__(self, min_samples: int = 500):
        self.min_samples = min_samples
        self.iso: IsotonicRegression | None = None

    def fit(self, p, y, w=None) -> "IsotonicCalibrator":
        if len(p) >= self.min_samples and len(np.unique(y)) == 2:
            self.iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y, sample_weight=w)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        return p if self.iso is None else self.iso.predict(p)
