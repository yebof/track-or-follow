"""Performance, prediction and calibration metrics with the inference used in the paper.

Formula sources are listed with each function; the corresponding papers must be in literature/ before
they are cited in the manuscript.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ------------------------------------------------------------------------------------ performance
def annualized_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0) if len(r) else float("nan")


def annualized_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.dropna().std(ddof=1) * math.sqrt(periods))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = (r - rf).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = (r - rf).dropna()
    dd = math.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(ex.mean() / dd * math.sqrt(periods)) if dd > 0 else float("nan")


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def calmar(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    nav = (1.0 + r.dropna()).cumprod()
    mdd = max_drawdown(nav)
    return float(annualized_return(r, periods) / abs(mdd)) if mdd < 0 else float("nan")


def summarize(r: pd.Series, rf: pd.Series | float = 0.0, benchmark: pd.Series | None = None) -> dict:
    nav = (1.0 + r.fillna(0.0)).cumprod()
    out = {
        "ann_return": annualized_return(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(nav),
        "calmar": calmar(r),
        "total_return": float(nav.iloc[-1] - 1.0),
        "n_days": int(r.notna().sum()),
    }
    if benchmark is not None:
        a = pd.concat([r, benchmark], axis=1).dropna()
        act = a.iloc[:, 0] - a.iloc[:, 1]
        te = act.std(ddof=1) * math.sqrt(TRADING_DAYS)
        out["tracking_error"] = float(te)
        out["information_ratio"] = float(act.mean() * TRADING_DAYS / te) if te > 0 else float("nan")
        beta = np.cov(a.iloc[:, 0], a.iloc[:, 1])[0, 1] / a.iloc[:, 1].var(ddof=1)
        out["beta"] = float(beta)
        out["alpha_ann"] = float((a.iloc[:, 0].mean() - beta * a.iloc[:, 1].mean()) * TRADING_DAYS)
    return out


# ------------------------------------------------------------------------------- signal quality
def information_coefficient(scores: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """Per-date cross-sectional correlation between scores and forward returns (both dates x symbols)."""
    out = {}
    for d in scores.index.intersection(fwd.index):
        a = pd.concat([scores.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(a) >= 5:
            out[d] = a.iloc[:, 0].corr(a.iloc[:, 1], method=method)
    return pd.Series(out, dtype=float)


def t_stat_newey_west(x: pd.Series, lags: int = 4) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) long-run variance."""
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    lrv = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(lrv / n)) if lrv > 0 else float("nan")


# ---------------------------------------------------------------------------------- inference
def block_bootstrap_ci(
    r: pd.Series, stat=sharpe, block: int = 20, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Circular moving-block bootstrap percentile CI for a statistic of a return series."""
    x = r.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        vals.append(stat(pd.Series(x[idx])))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sharpe_diff_bootstrap(
    r1: pd.Series, r2: pd.Series, block: int = 20, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired circular block bootstrap of SR(r1) - SR(r2): studentization-free percentile test in the
    spirit of Ledoit & Wolf (2008). Returns the difference, a 95% CI and a two-sided p-value."""
    a = pd.concat([r1, r2], axis=1).dropna().to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))

    def sr(x):
        s = x.std(ddof=1)
        return x.mean() / s * math.sqrt(TRADING_DAYS) if s > 0 else np.nan

    d0 = sr(a[:, 0]) - sr(a[:, 1])
    ds = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        ds.append(sr(a[idx, 0]) - sr(a[idx, 1]))
    ds = np.array(ds)
    centered = ds - ds.mean()
    p = float(np.mean(np.abs(centered) >= abs(d0)))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return {"diff": float(d0), "ci95": (float(lo), float(hi)), "p_value": p}


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado): P(true SR > benchmark) given skew/kurtosis; per-period SRs."""
    x = r.dropna()
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """DSR (Bailey & Lopez de Prado): PSR against the expected maximum SR of N independent trials,
    using per-period Sharpe ratios of all trials tried."""
    t = np.asarray(trial_sharpes, dtype=float)
    t = t[np.isfinite(t)]
    n_trials = len(t)
    if n_trials < 2:
        return probabilistic_sharpe(r, 0.0)
    emc = 0.5772156649
    v = t.var(ddof=1)
    sr0 = math.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials) + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, sr0)


# -------------------------------------------------------------------------------- calibration
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins over P(y=1)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "p_mean": p[m].mean(), "y_rate": y[m].mean()})
    return pd.DataFrame(rows)


# ================================================================ v1 additions (design §12.1, §13.1, §13.5-13.6)
# Every prediction-level test aggregates to a weekly series first and uses Newey-West(4); one-sided p-values use a
# t reference with n - 1 degrees of freedom (slightly conservative next to the normal at n ~ 115).
def _nw_p_one_sided(t: float, n: int) -> float:
    return float(stats.t.sf(t, df=max(n - 1, 1))) if np.isfinite(t) else float("nan")


def paired_ic_diff(ic_a: pd.Series, ic_b: pd.Series, lags: int = 4) -> dict:
    """C3 contrast: dIC_t = IC^A_t - IC^B_t over common weeks; Newey-West t; one-sided p for mean(dIC) > 0."""
    d = pd.concat([ic_a, ic_b], axis=1, join="inner").dropna()
    d = d.iloc[:, 0] - d.iloc[:, 1]
    t = t_stat_newey_west(d, lags)
    return {"mean": float(d.mean()) if len(d) else float("nan"), "t_nw": t, "p_one_sided": _nw_p_one_sided(t, len(d)),
            "n_weeks": int(len(d)), "lags": lags}


def dm_logloss(p_a, p_b, y, week, lags: int = 4, eps: float = 1e-4) -> dict:
    """Diebold-Mariano-style paired log-loss test. Per stock-week loss with p clipped to [eps, 1 - eps], weekly means
    L_t, d_t = L^B_t - L^A_t (positive: A has lower loss); Newey-West t; one-sided p for mean(d) > 0."""
    y = np.asarray(y, dtype=float)

    def loss(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    f = pd.DataFrame({"w": np.asarray(week), "la": loss(p_a), "lb": loss(p_b)}).groupby("w").mean()
    d = (f["lb"] - f["la"]).sort_index()
    t = t_stat_newey_west(d, lags)
    return {"mean_diff": float(d.mean()), "t_nw": t, "p_one_sided": _nw_p_one_sided(t, len(d)), "n_weeks": int(len(d)),
            "loss_a": float(f["la"].mean()), "loss_b": float(f["lb"].mean())}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control), in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def benjamini_hochberg(pvals, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up: (adjusted p-values, reject mask at FDR q), in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= q


def cer(r: pd.Series, gamma: float = 3.0, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Certainty-equivalent return CER_gamma = periods * (mean(r) - gamma * Var(r) / 2), as defined in singh2026ordinal
    (p.4; literature/text/singh2026ordinal.txt), here on returns in excess of `rf` (design §9.4)."""
    ex = (r - rf).dropna()
    return float(ex.mean() * periods - 0.5 * gamma * ex.var(ddof=1) * periods) if len(ex) > 1 else float("nan")


def cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """Expected shortfall at `alpha`, as a positive loss: minus the mean of returns at or below the (1-alpha) quantile."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    if not len(x):
        return float("nan")
    q = np.quantile(x, 1 - alpha)
    return float(-x[x <= q].mean())


def murphy_decomposition(p, y, n_bins: int | None = 10) -> dict:
    """Brier = reliability - resolution + uncertainty + within-bin variance - within-bin covariance (exact).
    `n_bins=None` groups by distinct forecast values, where the two within-bin terms vanish (useful for saturated
    emulator outputs with few distinct values)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    n = len(p)
    if n_bins is None:
        _, g = np.unique(p, return_inverse=True)
    else:
        g = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ybar = y.mean()
    rel = res = wbv = wbc = 0.0
    for k in np.unique(g):
        m = g == k
        pk, yk = p[m].mean(), y[m].mean()
        rel += m.sum() * (pk - yk) ** 2
        res += m.sum() * (yk - ybar) ** 2
        wbv += ((p[m] - pk) ** 2).sum()
        wbc += 2 * ((p[m] - pk) * (y[m] - yk)).sum()
    out = {"brier": float(np.mean((p - y) ** 2)), "reliability": rel / n, "resolution": res / n,
           "uncertainty": float(ybar * (1 - ybar)), "within_bin_variance": wbv / n, "within_bin_covariance": wbc / n}
    return {k: float(v) for k, v in out.items()}


def auc(scores, y) -> float:
    """ROC AUC via the Mann-Whitney statistic with average ranks for ties."""
    s, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    n1 = int((y == 1).sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def asr_cluster_bootstrap(success_a, cluster, success_b=None, n_boot: int = 10_000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Week-clustered bootstrap for attack success rates (C1). Units are resampled by cluster (week) with replacement.
    With `success_b`, the statistic is the paired difference ASR_a - ASR_b and `p_one_sided` tests ASR_a < ASR_b with
    the null-centered bootstrap distribution; without it, a CI for ASR_a."""
    a = np.asarray(success_a, dtype=float)
    b = None if success_b is None else np.asarray(success_b, dtype=float)
    codes, cl = np.unique(np.asarray(cluster), return_inverse=True)
    G = len(codes)
    n_c = np.bincount(cl, minlength=G).astype(float)
    sa = np.bincount(cl, weights=a, minlength=G)
    sb = np.bincount(cl, weights=b, minlength=G) if b is not None else np.zeros(G)
    est = a.mean() - (b.mean() if b is not None else 0.0)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    chunk = 1000
    for i in range(0, n_boot, chunk):
        idx = rng.integers(0, G, size=(min(chunk, n_boot - i), G))
        nn = n_c[idx].sum(axis=1)
        boot[i : i + len(idx)] = (sa[idx].sum(axis=1) - sb[idx].sum(axis=1)) / nn
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out = {"estimate": float(est), "ci": (float(lo), float(hi)), "n_units": int(len(a)), "n_clusters": int(G)}
    if b is not None:
        out["p_one_sided"] = float(np.mean(boot - est <= est))
        out["asr_a"], out["asr_b"] = float(a.mean()), float(b.mean())
    return out


def mcnemar(a, b) -> dict:
    """McNemar test on paired binary outcomes: exact binomial p on the discordant pairs (two-sided and one-sided for
    'a succeeds less often than b'), plus the continuity-corrected chi-square."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p_exact": 1.0, "p_one_sided": 1.0, "chi2": 0.0, "p_chi2": 1.0}
    chi2 = (abs(n10 - n01) - 1) ** 2 / n
    return {"n10": n10, "n01": n01, "p_exact": float(stats.binomtest(n10, n, 0.5).pvalue),
            "p_one_sided": float(stats.binomtest(n10, n, 0.5, alternative="less").pvalue),
            "chi2": float(chi2), "p_chi2": float(stats.chi2.sf(chi2, 1))}


# ================================================================ day-2 additions (design_v2_delta PE-2, PE-7, VC-1, VC-6, RB-4)
# ---------------------------------------------------------------- PE-2: drawdown family and break-even cost
def drawdown_series(r: pd.Series) -> pd.Series:
    """Percentage drawdown of the compounded NAV, nav / running max - 1 (<= 0), one value per period."""
    nav = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return nav / nav.cummax() - 1.0


def average_drawdown(r: pd.Series) -> float:
    """AvDD: mean drawdown magnitude over all periods (positive number). chekhlov2005drawdown define AvDD and CDD on
    the absolute drawdown of the uncompounded cumulative return (p.7-8); we use the percentage drawdown of the
    compounded NAV, the same basis as the reported maximum drawdown."""
    dd = -drawdown_series(r)
    return float(dd.mean()) if len(dd) else float("nan")


def conditional_drawdown(r: pd.Series, alpha: float = 0.8) -> float:
    """CDD_alpha: mean of the worst (1 - alpha) share of the drawdown values of one path (CVaR-style; alpha = 0 gives
    AvDD and alpha -> 1 the maximum drawdown; chekhlov2005drawdown Definitions 3.3-3.4, p.11, p.13). Positive number.
    alpha = 0.8 is the level design_v2_delta PE-2 names."""
    dd = np.sort(-drawdown_series(r).to_numpy(dtype=float))[::-1]
    if not len(dd):
        return float("nan")
    k = max(1, int(math.ceil((1.0 - alpha) * len(dd) - 1e-12)))
    return float(dd[:k].mean())


def break_even_cost(r: pd.Series, turnover: pd.Series, r_bench: pd.Series, turnover_bench: pd.Series,
                    cost_bps: float) -> float:
    """Per-side cost (bps) at which the book's mean daily return equals the benchmark's (PE-2: the cost at which the
    excess return vs EW-U100 is zero). `r`, `r_bench`: daily returns net of `cost_bps`; `turnover*`: traded notional
    sum |dw| per fill (backtest.engine BacktestResult.turnover, the base the engine charges costs on). To first order
    each fill day's return moves by (c0 - c) / 1e4 x traded notional, so
        c* = c0 + 1e4 x n x (mean r - mean r_bench) / (sum traded - sum traded_bench)
    over the n common days. +inf: the book trades no more than the benchmark and is ahead at any cost; -inf: it trades
    no more and is behind."""
    a = pd.concat([pd.Series(r), pd.Series(r_bench)], axis=1, join="inner").dropna()
    if a.empty:
        return float("nan")
    days = a.index
    dm = float(a.iloc[:, 0].mean() - a.iloc[:, 1].mean())
    to = float(pd.Series(turnover).reindex(days).fillna(0.0).sum())
    tob = float(pd.Series(turnover_bench).reindex(days).fillna(0.0).sum())
    dto = to - tob
    if dto <= 0:
        return float("inf") if dm >= 0 else float("-inf")
    return float(cost_bps + 1e4 * len(days) * dm / dto)


# ---------------------------------------------------------------- PE-7: Ledoit-Wolf studentized block bootstrap
def _lw_v(x: np.ndarray) -> np.ndarray:
    """(mu1, mu2, gamma1, gamma2) of an (T, 2) or (M, T, 2) array of return pairs."""
    return np.concatenate([x.mean(axis=-2), (x**2).mean(axis=-2)], axis=-1)


def _lw_delta(v: np.ndarray) -> np.ndarray:
    a, b, c, d = (v[..., k] for k in range(4))
    return a / np.sqrt(c - a**2) - b / np.sqrt(d - b**2)


def _lw_grad(v: np.ndarray) -> np.ndarray:
    a, b, c, d = (v[..., k] for k in range(4))
    return np.stack([c / (c - a**2) ** 1.5, -d / (d - b**2) ** 1.5,
                     -0.5 * a / (c - a**2) ** 1.5, 0.5 * b / (d - b**2) ** 1.5], axis=-1)


def _qs(x: np.ndarray) -> np.ndarray:
    z = 6.0 * np.pi * x / 5.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 25.0 / (12.0 * np.pi**2 * x**2) * (np.sin(z) / z - np.cos(z))
    return np.where(x == 0, 1.0, k)


def hac_prewhitened_qs(y: np.ndarray) -> np.ndarray:
    """Psi-hat for centered vectors y (T, k): VAR(1) prewhitening, quadratic-spectral kernel with the Andrews (1991)
    AR(1) plug-in bandwidth, recoloring, and the T/(T - 4) factor, as ledoit2008robust prescribe for s(Delta-hat)
    (p.3-5: 'the prewhitened QS kernel of Andrews and Monahan (1992)')."""
    T, k = y.shape
    y0, y1 = y[:-1], y[1:]
    A = np.linalg.lstsq(y0, y1, rcond=None)[0].T  # y_t = A y_{t-1} + e_t
    e = y1 - y0 @ A.T
    n = len(e)
    num = den = 0.0
    for j in range(k):
        s = e[:, j]
        rho = float(np.dot(s[1:], s[:-1]) / np.dot(s[:-1], s[:-1])) if np.dot(s[:-1], s[:-1]) > 0 else 0.0
        rho = min(max(rho, -0.97), 0.97)
        sig2 = float(np.mean((s[1:] - rho * s[:-1]) ** 2))
        num += 4 * rho**2 * sig2**2 / (1 - rho) ** 8
        den += sig2**2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    S = 1.3221 * (alpha2 * n) ** 0.2 if alpha2 > 0 else 1.0
    psi = e.T @ e / n
    for j in range(1, n):  # the QS kernel has no truncation point: every lag gets its (decaying) weight
        w = float(_qs(np.array(j / S)))
        g = e[j:].T @ e[:-j] / n
        psi += w * (g + g.T)
    B = np.linalg.inv(np.eye(k) - A)
    return B @ psi @ B.T * T / (T - 4)


def sharpe_diff_lw(r1: pd.Series, r2: pd.Series, rf: pd.Series | float = 0.0, block: int = 5, n_boot: int = 4999,
                   seed: int = 0, alpha: float = 0.05, periods: int = TRADING_DAYS) -> dict:
    """PE-7: H0 SR(r1) = SR(r2) by the studentized circular block bootstrap of ledoit2008robust (Section 3.2.2,
    p.4-6): Delta = mu1/sd1 - mu2/sd2 on excess returns, s(Delta-hat) from the prewhitened-QS HAC Psi-hat, the 'natural'
    block standard error in the bootstrap world, a symmetric studentized CI (Eq. 7) and the p-value of Eq. (9).
    `block` is the block size b (calibrate_block_size implements their Algorithm 3.1)."""
    ex = pd.concat([pd.Series(r1) - rf, pd.Series(r2) - rf], axis=1, join="inner").dropna().to_numpy(dtype=float)
    T = len(ex)
    if T < 3 * block or T < 10:
        return {"diff": float("nan"), "p_value": float("nan"), "n": T}
    v = _lw_v(ex)
    d_hat = float(_lw_delta(v))
    g = _lw_grad(v)
    psi = hac_prewhitened_qs(np.column_stack([ex - v[:2], ex**2 - v[2:]]))
    se = float(np.sqrt(g @ psi @ g / T))
    rng = np.random.default_rng(seed)
    b = int(block)
    nb = T // b
    Tb = nb * b
    stats_ = np.empty(n_boot)
    chunk = 500
    for i in range(0, n_boot, chunk):
        m = min(chunk, n_boot - i)
        starts = rng.integers(0, T, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % T).reshape(m, Tb)
        xs = ex[idx]  # (m, Tb, 2)
        vs = _lw_v(xs)
        ys = np.concatenate([xs - vs[:, None, :2], xs**2 - vs[:, None, 2:]], axis=2)  # (m, Tb, 4)
        f = ys.reshape(m, nb, b, 4).sum(axis=2) / np.sqrt(b)  # (m, nb, 4)
        psis = np.einsum("mjk,mjl->mkl", f, f) / nb
        gs = _lw_grad(vs)
        ses = np.sqrt(np.einsum("mk,mkl,ml->m", gs, psis, gs) / Tb)
        stats_[i : i + m] = np.abs(_lw_delta(vs) - d_hat) / ses
    d = abs(d_hat) / se if se > 0 else float("inf")
    stats_ = stats_[np.isfinite(stats_)]
    zq = float(np.quantile(stats_, 1 - alpha)) if len(stats_) else float("nan")
    return {"diff": d_hat, "diff_annualized": d_hat * math.sqrt(periods), "se": se, "stat": d,
            "p_value": float((np.sum(stats_ >= d) + 1) / (len(stats_) + 1)), "ci": (d_hat - zq * se, d_hat + zq * se),
            "ci_annualized": ((d_hat - zq * se) * math.sqrt(periods), (d_hat + zq * se) * math.sqrt(periods)),
            "block": b, "n_boot": n_boot, "n": T, "sr1": float(v[0] / np.sqrt(v[2] - v[0] ** 2)),
            "sr2": float(v[1] / np.sqrt(v[3] - v[1] ** 2))}


def _stationary_bootstrap(e: np.ndarray, n: int, mean_block: float, rng) -> np.ndarray:
    T = len(e)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(T)
    new = rng.uniform(size=n) < 1.0 / mean_block
    starts = rng.integers(T, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new[t] else (idx[t - 1] + 1) % T
    return e[idx]


def calibrate_block_size(r1: pd.Series, r2: pd.Series, rf: pd.Series | float = 0.0,
                         blocks: tuple[int, ...] = (1, 2, 4, 6, 8, 10), n_pseudo: int = 200, n_boot: int = 199,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """ledoit2008robust Algorithm 3.1 (p.5): fit a VAR(1) to the return pairs, generate `n_pseudo` sequences from it
    with stationary-bootstrap residuals (mean block 5, their simulation choice, p.6), compute the bootstrap CI for
    each block size, and pick the size whose empirical coverage of Delta-hat is closest to 1 - alpha."""
    ex = pd.concat([pd.Series(r1) - rf, pd.Series(r2) - rf], axis=1, join="inner").dropna().to_numpy(dtype=float)
    T = len(ex)
    mu = ex.mean(axis=0)
    y0, y1 = ex[:-1] - mu, ex[1:] - mu
    A = np.linalg.lstsq(y0, y1, rcond=None)[0].T
    e = y1 - y0 @ A.T
    d_true = float(_lw_delta(_lw_v(ex)))
    rng = np.random.default_rng(seed)
    cover = {b: 0 for b in blocks}
    for k in range(n_pseudo):
        eps = _stationary_bootstrap(e, T + 50, 5.0, rng)
        y = np.zeros((T + 50, 2))
        for t in range(1, T + 50):
            y[t] = A @ y[t - 1] + eps[t]
        x = pd.DataFrame(y[50:] + mu)
        for b in blocks:
            res = sharpe_diff_lw(x[0], x[1], 0.0, block=b, n_boot=n_boot, seed=seed + 1 + k, alpha=alpha)
            lo, hi = res.get("ci", (np.nan, np.nan))
            cover[b] += int(lo <= d_true <= hi)
    cov = {b: c / n_pseudo for b, c in cover.items()}
    best = min(blocks, key=lambda b: (abs(cov[b] - (1 - alpha)), b))
    return {"block": best, "coverage": cov, "n_pseudo": n_pseudo, "n_boot": n_boot}


# ---------------------------------------------------------------- VC-1 / VC-6: reference forecasters, scoring hygiene
LOGLOSS_EPS = 1e-4  # probabilities are clipped to [eps, 1 - eps] before the log (stated with every table)
ECE_BINS = 10  # equal-width bins over P(y = 1)


def score_table(p, y, eps: float = LOGLOSS_EPS, n_bins: int = ECE_BINS) -> dict:
    """VC-6 scoring hygiene in one place: binary Brier; log loss with p clipped to [eps, 1 - eps]; ECE with
    `n_bins` equal-width bins; the Murphy decomposition (reliability, resolution, uncertainty; resolution first, VC-4)
    on the same bins; AUC. The settings are returned with the numbers."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(p):
        return {"n": 0}
    mu = murphy_decomposition(p, y, n_bins)
    return {"n": int(len(p)), "resolution": mu["resolution"], "uncertainty": mu["uncertainty"],
            "reliability": mu["reliability"], "auc": auc(p, y), "brier": brier(p, y),
            "log_loss": log_loss(p, y, eps), "ece": ece(p, y, n_bins), "eps": eps, "n_bins": n_bins}


def skill_scores(p, y, p_ref, eps: float = LOGLOSS_EPS) -> dict:
    """Skill against a reference forecaster (VC-1): 1 - S / S_ref for log loss and Brier (positive = better)."""
    p, y, q = (np.asarray(a, dtype=float) for a in (p, y, p_ref))
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(q)
    p, y, q = p[ok], y[ok], q[ok]
    if not len(p):
        return {"n": 0}
    ll, llr = log_loss(p, y, eps), log_loss(q, y, eps)
    bs, bsr = brier(p, y), brier(q, y)
    return {"n": int(len(p)), "log_loss_skill": 1.0 - ll / llr if llr > 0 else float("nan"),
            "brier_skill": 1.0 - bs / bsr if bsr > 0 else float("nan"), "log_loss_ref": llr, "brier_ref": bsr}


def trailing_base_rate(labels, ts) -> np.ndarray:
    """VC-1 reference forecaster: at decision t, the share of y = 1 among the labels released by
    `LabelStore.mature(c_t)` (0.5 before any label matures). One value per entry of `ts`."""
    from jevtrader.data.news import decision_cutoff

    ts = pd.to_datetime(pd.Series(ts))
    rate = {}
    for t in ts.unique():
        m = labels.mature(decision_cutoff(pd.Timestamp(t)))["y"]
        rate[pd.Timestamp(t)] = float(m.mean()) if len(m) else 0.5
    return ts.map(rate).to_numpy(dtype=float)


def restrict_to(rows: pd.DataFrame, reference: pd.DataFrame, keys=("t", "symbol")) -> pd.DataFrame:
    """VC-6 / PE-11: rows of a U100 table restricted to the stock-weeks of a subset run (S50/S30), so a subset
    baseline is only ever compared with the same names and weeks."""
    k = reference[list(keys)].drop_duplicates()
    return rows.merge(k, on=list(keys), how="inner")


# ---------------------------------------------------------------- RB-4: signed harm and the AgentDojo triad
def signed_harm(w_clean: pd.Series, w_attacked: pd.Series, r: pd.Series, planted: str) -> dict:
    """Paired change of one book's holding-week return caused by one attacked unit (attacked - clean; negative =
    harm), in bps of NAV, decomposed as dw_planted x r_planted + the rest. With a fully invested swap the rest is
    -dw_planted x r_displaced, where r_displaced is the dw-weighted return of the names the planted name displaced.
    `r`: holding-week returns (LabelStore `r`, open(e_t) to open(e_{t+1})); a name without a return counts 0."""
    names = sorted(set(w_clean.index) | set(w_attacked.index))
    dw = pd.Series(w_attacked, dtype=float).reindex(names).fillna(0.0) - pd.Series(w_clean, dtype=float).reindex(names).fillna(0.0)
    rr = pd.Series(r, dtype=float).reindex(names)
    missing = [s for s in names if dw[s] != 0 and not np.isfinite(rr[s])]
    rr = rr.fillna(0.0)
    total = float((dw * rr).sum())
    dwp = float(dw.get(planted, 0.0))
    part = dwp * float(rr.get(planted, 0.0))
    rest = total - part
    out_names = dw[(dw < 0)]
    r_disp = float((-out_names * rr[out_names.index]).sum() / -out_names.sum()) if out_names.sum() < 0 else float("nan")
    return {"d_ret_bps": 1e4 * total, "planted_bps": 1e4 * part, "rest_bps": 1e4 * rest, "dw_planted": dwp,
            "r_planted": float(rr.get(planted, 0.0)), "r_displaced": r_disp, "entered": dwp > 0,
            "missing_returns": missing}


def harm_summary(d_bps) -> dict:
    """Mean, 5% CVaR (mean of the worst 5% of paired differences; signed, negative = loss) and sum, in bps of NAV."""
    x = np.asarray(d_bps, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    k = max(1, int(math.ceil(0.05 * len(x))))
    return {"n": int(len(x)), "mean_bps": float(x.mean()), "cvar5_bps": float(np.sort(x)[:k].mean()),
            "sum_bps": float(x.sum())}


def agentdojo_triad(ic_clean: pd.Series, ic_attacked: pd.Series, success) -> dict:
    """RB-4 triad in the spirit of debenedetti2024agentdojo: benign utility (clean weekly IC mean and IC-IR), utility
    under attack (the weekly IC of the attacked runs over the attacked weeks) and the targeted ASR."""
    def util(ic):
        ic = pd.Series(ic, dtype=float).dropna()
        sd = ic.std(ddof=1) if len(ic) > 1 else float("nan")
        return {"ic_mean": float(ic.mean()) if len(ic) else float("nan"),
                "ic_ir": float(ic.mean() / sd) if len(ic) > 1 and sd > 0 else float("nan"), "n_weeks": int(len(ic))}

    s = np.asarray(success, dtype=float)
    s = s[np.isfinite(s)]
    return {"benign_utility": util(ic_clean), "utility_under_attack": util(ic_attacked),
            "targeted_asr": float(s.mean()) if len(s) else float("nan"), "n_units": int(len(s))}
