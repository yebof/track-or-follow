"""Online MAP logistic regression with a skeptical Gaussian prior (design §6.2-6.3).

    beta_hat = argmin_beta  sum_i w_i * logloss(y_i, o_i + x_i . beta)  +  ||beta - mu||^2 / (2 sigma^2)

The loss is a weighted *sum*, not a mean, so the prior's strength relative to the data is explicit (validity m13):
with standardized features and p ~ 1/2 each row adds Fisher information ~ w/4, so sigma = 0.04 is worth about
2,500 unit-weight rows per feature. `w` carries the half-life decay on label `available_at`. The model is refit from
the prior at every decision (no warm start), so a fit depends only on the rows it is given, which is what makes
decisions replayable bit-for-bit. The Laplace covariance (inverse Hessian at the mode) is stored with the snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def decay_weights(age_weeks: np.ndarray, half_life_weeks: float | None) -> np.ndarray:
    """w = 0.5 ** (age / H); H = None or inf gives equal weights (V-nodecay)."""
    a = np.asarray(age_weeks, dtype=float)
    if half_life_weeks is None or not np.isfinite(half_life_weeks):
        return np.ones_like(a)
    return 0.5 ** (a / float(half_life_weeks))


@dataclass
class OnlineLogit:
    prior_sd: float | np.ndarray = 0.04
    prior_mean: np.ndarray | None = None
    coef_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cov_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_: int = 0
    sum_w_: float = 0.0
    converged_: bool = True

    def fit(self, X, y, w=None, prior_sd=None, offset=None) -> OnlineLogit:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        n, k = X.shape
        y = np.asarray(y, dtype=float)
        w = np.ones(n) if w is None else np.asarray(w, dtype=float)
        o = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
        if prior_sd is not None:
            self.prior_sd = prior_sd
        sd = np.broadcast_to(np.asarray(self.prior_sd, dtype=float), (k,)).copy()
        mu = np.zeros(k) if self.prior_mean is None else np.asarray(self.prior_mean, dtype=float)
        prec = 1.0 / sd**2
        if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(w).all() and np.isfinite(o).all()):
            raise ValueError("non-finite inputs to OnlineLogit.fit")
        self.n_, self.sum_w_ = n, float(w.sum())
        if n == 0 or k == 0:
            self.coef_, self.cov_, self.converged_ = mu.copy(), np.diag(1.0 / prec), True
            return self

        def f(b):
            z = o + X @ b
            nll = -(w * (y * log_expit(z) + (1.0 - y) * log_expit(-z))).sum()
            d = b - mu
            g = X.T @ (w * (expit(z) - y)) + prec * d
            return nll + 0.5 * (prec * d * d).sum(), g

        res = minimize(f, x0=mu.copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-15})
        b = res.x
        for _ in range(4):  # Newton polish: the problem is strictly convex and small (k < 50)
            z = o + X @ b
            p = expit(z)
            g = X.T @ (w * (p - y)) + prec * (b - mu)
            H = (X * (w * p * (1.0 - p))[:, None]).T @ X + np.diag(prec)
            step = np.linalg.solve(H, g)
            b = b - step
            if np.max(np.abs(step)) < 1e-13:
                break
        p = expit(o + X @ b)
        H = (X * (w * p * (1.0 - p))[:, None]).T @ X + np.diag(prec)
        self.coef_, self.cov_ = b, np.linalg.inv(H)
        self.converged_ = bool(np.max(np.abs(X.T @ (w * (p - y)) + prec * (b - mu))) < 1e-6 * max(1.0, self.sum_w_))
        return self

    def logit(self, X, offset=None) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.coef_
        return z if offset is None else z + np.asarray(offset, dtype=float)

    def predict(self, X, offset=None) -> np.ndarray:
        return expit(self.logit(X, offset))
