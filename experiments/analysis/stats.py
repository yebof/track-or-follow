"""Statistics of the pre-registered analyses that src/jevtrader/eval/metrics.py does not already provide.

Everything here is a pure function of arrays (no file, no lock), tested on synthetic data with known answers in
tests/test_analysis_stats.py. Where metrics.py has the function (Holm, the week-clustered ASR bootstrap, the paired
IC difference, the Diebold-Mariano log-loss test, AUC, the score table, the DSR), the entry points call metrics.py
directly; the functions below add the confidence intervals, the one-sample Newey-West test with a CI, the weekly
Spearman series, week-block bootstraps, the studentized block-bootstrap CI of one Sharpe ratio, the DSR with the
correlation-implied number of trials, TOST, Cohen's kappa and a few intervals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

NW_LAGS = 4  # prereg §3.2-§3.3: every weekly series uses Newey-West(4)
LEVEL = 0.95


# ------------------------------------------------------------------------------------------------ Newey-West
def nw_lrv(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Bartlett long-run variance of a centered series, as eval.metrics.t_stat_newey_west."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    lrv = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(lrv)


def nw_test(x, lags: int = NW_LAGS, alternative: str = "greater", level: float = LEVEL) -> dict:
    """Test of mean(x) = 0 with a Newey-West standard error. The p-value uses a t reference with n - 1 degrees of
    freedom, as eval.metrics (slightly conservative next to the normal); `alternative` in {greater, less,
    two-sided}. The CI is mean +/- t_{(1+level)/2, n-1} x se."""
    s = pd.Series(x, dtype=float).dropna()
    v = s.to_numpy()
    n = len(v)
    out = {"n": n, "lags": lags, "alternative": alternative, "level": level}
    if n < 3:
        return {**out, "mean": float(v.mean()) if n else float("nan"), "se": float("nan"), "t": float("nan"),
                "p": float("nan"), "ci": [float("nan"), float("nan")]}
    lrv = nw_lrv(v, lags)
    se = math.sqrt(lrv / n) if lrv > 0 else float("nan")
    m = float(v.mean())
    t = m / se if se and np.isfinite(se) else float("nan")
    df = n - 1
    if not np.isfinite(t):
        p = float("nan")
    elif alternative == "greater":
        p = float(stats.t.sf(t, df))
    elif alternative == "less":
        p = float(stats.t.cdf(t, df))
    else:
        p = float(2 * stats.t.sf(abs(t), df))
    q = float(stats.t.ppf(0.5 + level / 2, df))
    return {**out, "mean": m, "se": se, "t": t, "df": df, "p": p,
            "ci": [m - q * se, m + q * se] if np.isfinite(se) else [float("nan"), float("nan")]}


def tost_nw(d, delta: float, lags: int = NW_LAGS, alpha: float = 0.05) -> dict:
    """Two one-sided NW tests of |mean(d)| < delta (prereg §4.3 parity rule): parity may be stated only if both
    one-sided nulls (mean <= -delta, mean >= delta) are rejected at alpha."""
    s = pd.Series(d, dtype=float).dropna()
    lo = nw_test(s + delta, lags, "greater")
    hi = nw_test(s - delta, lags, "less")
    p = max(lo["p"], hi["p"]) if np.isfinite(lo["p"]) and np.isfinite(hi["p"]) else float("nan")
    return {"delta": delta, "alpha": alpha, "mean": lo["mean"] - delta if np.isfinite(lo["mean"]) else float("nan"),
            "p_lower": lo["p"], "p_upper": hi["p"], "p": p, "parity": bool(np.isfinite(p) and p < alpha),
            "n": lo["n"]}


# ------------------------------------------------------------------------------------------------ weekly series
def spearman(a, b) -> float:
    """Spearman correlation with average ranks for ties (scipy); NaN if either side is constant or n < 3."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def weekly_spearman(frame: pd.DataFrame, x: str, y: str, week: str = "t", min_n: int = 4) -> pd.DataFrame:
    """One row per week: n (finite pairs) and the Spearman correlation of x with y over that week's rows; weeks
    with fewer than `min_n` pairs or a constant side have rho NaN."""
    rows = []
    for w, g in frame.groupby(week, sort=True):
        ok = np.isfinite(g[x].to_numpy(dtype=float)) & np.isfinite(g[y].to_numpy(dtype=float))
        n = int(ok.sum())
        rho = spearman(g[x].to_numpy(dtype=float)[ok], g[y].to_numpy(dtype=float)[ok]) if n >= min_n else float("nan")
        rows.append({week: pd.Timestamp(w), "n": n, "rho": rho})
    return pd.DataFrame(rows, columns=[week, "n", "rho"])


def weekly_mean_test(series: pd.DataFrame, value: str = "rho", lags: int = NW_LAGS,
                     alternative: str = "greater") -> dict:
    """The mean of a weekly statistic over the weeks where it is defined, with its NW(lags) test and CI."""
    s = series.dropna(subset=[value]).sort_values(series.columns[0])
    out = nw_test(s[value].to_numpy(), lags, alternative)
    out["n_weeks"] = len(s)
    out["n_weeks_undefined"] = int(series[value].isna().sum())
    if "n" in s:
        out["n_units"] = int(s["n"].sum())
    return out


# ------------------------------------------------------------------------------------------------ bootstraps
def week_block_bootstrap(frame: pd.DataFrame, week: str, stat: Callable[[pd.DataFrame], float], n_boot: int = 2000,
                         seed: int = 0, level: float = LEVEL) -> dict:
    """Percentile CI of `stat(frame)` resampling whole weeks with replacement (the week is the block)."""
    groups = [g for _, g in frame.groupby(week, sort=True)]
    est = float(stat(frame))
    G = len(groups)
    if G < 2:
        return {"estimate": est, "ci": [float("nan"), float("nan")], "n_weeks": G, "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        vals[b] = stat(pd.concat([groups[i] for i in idx], ignore_index=True))
    if not np.isfinite(vals).any():
        return {"estimate": est, "ci": [float("nan"), float("nan")], "n_weeks": G, "n_boot": n_boot, "seed": seed}
    lo, hi = np.nanpercentile(vals, [50 * (1 - level), 50 * (1 + level)])
    return {"estimate": est, "ci": [float(lo), float(hi)], "n_weeks": G, "n_boot": n_boot, "seed": seed}


def week_block_bootstrap_xy(frame: pd.DataFrame, week: str, x: str, y: str,
                            stat: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 2000, seed: int = 0,
                            level: float = LEVEL) -> dict:
    """`week_block_bootstrap` for a statistic of two columns (a correlation, an AUC): the same resampling of whole
    weeks with replacement and the same random stream, on numpy arrays (no DataFrame per replicate)."""
    groups = [(g[x].to_numpy(dtype=float), g[y].to_numpy(dtype=float)) for _, g in frame.groupby(week, sort=True)]
    est = float(stat(frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)))
    G = len(groups)
    if G < 2:
        return {"estimate": est, "ci": [float("nan"), float("nan")], "n_weeks": G, "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        vals[b] = stat(np.concatenate([groups[i][0] for i in idx]), np.concatenate([groups[i][1] for i in idx]))
    if not np.isfinite(vals).any():
        return {"estimate": est, "ci": [float("nan"), float("nan")], "n_weeks": G, "n_boot": n_boot, "seed": seed}
    lo, hi = np.nanpercentile(vals, [50 * (1 - level), 50 * (1 + level)])
    return {"estimate": est, "ci": [float(lo), float(hi)], "n_weeks": G, "n_boot": n_boot, "seed": seed}


def _pairwise_auc(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC of each row of `p` (B x n, NaN = missing) against the fixed labels `y` (n; NaN = missing) by the
    Mann-Whitney pair count (ties count 1/2), the same value as eval.metrics.auc row by row."""
    yv = np.asarray(y, dtype=float)
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    a, b = p[:, pos][:, :, None], p[:, neg][:, None, :]
    ok = np.isfinite(a) & np.isfinite(b)
    wins = ((a > b) + 0.5 * (a == b)) * ok
    n_pairs = ok.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_pairs > 0, wins.sum(axis=(1, 2)) / n_pairs, np.nan)


def name_permutation_auc(frame: pd.DataFrame, week: str, pos: str, x: str, y: str, n_perm: int = 10_000,
                         seed: int = 0, center: float = 0.5) -> dict:
    """Two-sided permutation test of AUC(x, y) = `center` for a panel of weeks x name positions (D1, prereg §7.2).

    One permutation draws a single permutation of the name positions and applies it in EVERY week: the x of position
    k is paired with the y of position pi(k) of the same week. It keeps each week's cross-section and each name's
    persistence over the weeks (the probe asks the same hash-chosen names every week), and breaks only the pairing of
    x-series with y-series, so it is exact under the null that the scores carry no information about the realized
    labels whenever the names are exchangeable under that null, even with 6 weeks (a percentile bootstrap over 6 week
    clusters is too narrow: audit 2026-09-28). The statistic is |AUC - center|; p = (1 + #{T_b >= T}) / (1 + B).
    Rows need `week`, integer `pos` (0..K-1 within a week), `x` and `y` (NaN allowed, dropped pairwise)."""
    g = frame[[week, pos, x, y]].copy()
    weeks = sorted(g[week].unique())
    K = int(g[pos].max()) + 1 if len(g) else 0
    wi = {w: i for i, w in enumerate(weeks)}
    X = np.full((len(weeks), K), np.nan)
    Y = np.full((len(weeks), K), np.nan)
    for w, k, xv, yv in g[[week, pos, x, y]].itertuples(index=False):
        X[wi[w], int(k)], Y[wi[w], int(k)] = float(xv), float(yv)
    yflat = Y.ravel()
    est = float(_pairwise_auc(X.ravel()[None, :], yflat)[0]) if K else float("nan")
    out = {"auc": est, "n_weeks": len(weeks), "n_positions": K, "n_perm": n_perm, "seed": seed,
           "statistic": f"|AUC - {center}|", "p_two_sided": float("nan")}
    if not np.isfinite(est) or K < 2:
        return out
    t_obs = abs(est - center)
    rng = np.random.default_rng(seed)
    exceed, done, chunk = 0, 0, 500
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = np.argsort(rng.random((m, K)), axis=1)  # one permutation of the positions per draw
        Xp = X[:, perms].transpose(1, 0, 2).reshape(m, -1)  # (m, weeks * K): x of position pi(k) in each week
        tb = np.abs(_pairwise_auc(Xp, yflat) - center)
        exceed += int(np.sum(tb >= t_obs - 1e-12))
        done += m
    out["p_two_sided"] = (1 + exceed) / (1 + n_perm)
    return out


def paired_block_bootstrap_mean(x: pd.Series, n_boot: int = 4999, block: int = 5, seed: int = 0,
                                level: float = LEVEL) -> dict:
    """Circular moving-block bootstrap percentile CI of a mean (e.g. a daily active-return difference)."""
    v = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(v)
    if n < 2 * block:
        return {"estimate": float(v.mean()) if n else float("nan"), "ci": [float("nan"), float("nan")], "n": n}
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return {"estimate": float(v.mean()), "ci": [float(lo), float(hi)], "n": n, "block": block, "n_boot": n_boot,
            "seed": seed}


# ------------------------------------------------------------------------------------------------ Sharpe
def sharpe_ci(excess, block: int = 5, n_boot: int = 4999, seed: int = 0, level: float = LEVEL,
              periods: int = 252) -> dict:
    """Sharpe ratio of one excess-return series with a studentized circular block-bootstrap CI: the one-series
    version of eval.metrics.sharpe_diff_lw (ledoit2008robust §3.2.2): SR = mu / sqrt(g - mu^2) with (mu, g) the
    first two raw moments, its delta-method s.e. from the prewhitened-QS HAC matrix of (x - mu, x^2 - g), the
    'natural' block s.e. in the bootstrap world, and the symmetric studentized interval SR +/- z* s.e.
    Per-period values plus the annualized ones (x sqrt(periods))."""
    from jevtrader.eval.metrics import hac_prewhitened_qs

    x = pd.Series(excess, dtype=float).dropna().to_numpy()
    T = len(x)
    if T < max(10, 3 * block):
        return {"sharpe": float("nan"), "ci": [float("nan"), float("nan")], "n": T}
    mu, g = x.mean(), (x**2).mean()
    var = g - mu**2
    sr = mu / math.sqrt(var)
    grad = np.array([g / var**1.5, -0.5 * mu / var**1.5])
    y = np.column_stack([x - mu, x**2 - g])
    psi = hac_prewhitened_qs(y)
    se = float(math.sqrt(max(grad @ psi @ grad, 0.0) / T))
    rng = np.random.default_rng(seed)
    nb = T // block
    Tb = nb * block
    stat = np.empty(n_boot)
    chunk = 500
    for i in range(0, n_boot, chunk):
        m = min(chunk, n_boot - i)
        starts = rng.integers(0, T, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(m, Tb)
        xs = x[idx]
        mus, gs = xs.mean(axis=1), (xs**2).mean(axis=1)
        vs = gs - mus**2
        srs = mus / np.sqrt(vs)
        ys = np.stack([xs - mus[:, None], xs**2 - gs[:, None]], axis=2)  # (m, Tb, 2)
        f = ys.reshape(m, nb, block, 2).sum(axis=2) / math.sqrt(block)
        ps = np.einsum("mjk,mjl->mkl", f, f) / nb
        gr = np.stack([gs / vs**1.5, -0.5 * mus / vs**1.5], axis=1)
        ses = np.sqrt(np.einsum("mk,mkl,ml->m", gr, ps, gr) / Tb)
        stat[i:i + m] = np.abs(srs - sr) / ses
    stat = stat[np.isfinite(stat)]
    zq = float(np.quantile(stat, level)) if len(stat) else float("nan")
    # the moments above use 1/T (the delta method's convention); the reported Sharpe uses the sample sd (ddof = 1)
    # like eval.metrics.sharpe, so the estimate and its interval are both scaled by sqrt((T - 1) / T)
    f = math.sqrt((T - 1) / T)
    a = math.sqrt(periods) * f
    return {"sharpe": float(sr * a), "ci": [float((sr - zq * se) * a), float((sr + zq * se) * a)],
            "sharpe_per_period": float(sr * f), "se_per_period": se * f,
            "z_star": zq, "n": T, "block": block, "n_boot": n_boot, "seed": seed, "level": level,
            "method": "studentized circular block bootstrap (ledoit2008robust, one series), ddof-1 scale"}


def expected_max_sharpe(var_sr: float, n_trials: float, mean_sr: float = 0.0) -> float:
    """bailey2014deflated Eq. (1): E[max SR] of n_trials Normal trials (the SR_0 of the DSR)."""
    emc = 0.5772156649
    n = float(n_trials)
    if n < 2 or not np.isfinite(var_sr) or var_sr <= 0:
        return float(mean_sr)
    return float(mean_sr + math.sqrt(var_sr) * ((1 - emc) * stats.norm.ppf(1 - 1 / n)
                                               + emc * stats.norm.ppf(1 - 1 / (n * math.e))))


def deflated_sharpe(excess, trial_srs: Sequence[float], n_trials: float | None = None) -> dict:
    """DSR (bailey2014deflated): PSR of the book's per-period SR against SR_0 = E[max] of `n_trials` trials whose SRs
    have the variance of `trial_srs` (per period). n_trials = None uses M = len(trial_srs), which reproduces
    eval.metrics.deflated_sharpe; pass N-hat for the correlation-implied count (Eq. 9, p.14)."""
    from jevtrader.eval.metrics import probabilistic_sharpe

    t = np.asarray(trial_srs, dtype=float)
    t = t[np.isfinite(t)]
    M = len(t)
    n = float(M if n_trials is None else n_trials)
    x = pd.Series(excess, dtype=float).dropna()
    if M < 2 or n < 2:
        return {"dsr": probabilistic_sharpe(x, 0.0), "sr0_per_period": 0.0, "n_trials": n, "M": M}
    sr0 = expected_max_sharpe(float(t.var(ddof=1)), n)
    return {"dsr": probabilistic_sharpe(x, sr0), "sr0_per_period": sr0, "n_trials": n, "M": M,
            "var_trial_sr": float(t.var(ddof=1)), "skew": float(stats.skew(x)),
            "kurtosis": float(stats.kurtosis(x, fisher=False)), "n_periods": len(x)}


def implied_independent_trials(returns: pd.DataFrame, M: int) -> dict:
    """bailey2014deflated Appendix A.3: the equal-weighted average off-diagonal correlation rho-bar of the trial
    return series (Eq. 8) and N-hat = rho-bar + (1 - rho-bar) M (Eq. 9, p.14). `returns`: one column per series with
    stored returns (the registry keeps summaries only); M: all registered trials."""
    r = returns.dropna(how="all")
    k = r.shape[1]
    if k < 2:
        return {"rho_bar": float("nan"), "n_hat": float(M), "n_series": k, "M": M}
    c = r.corr().to_numpy()
    off = c[~np.eye(k, dtype=bool)]
    rho = float(np.nanmean(off))
    return {"rho_bar": rho, "n_hat": float(rho + (1 - rho) * M), "n_series": k, "M": M}


# ------------------------------------------------------------------------------------------------ agreement
def cohen_kappa(a: Sequence, b: Sequence) -> dict:
    """Cohen's kappa of two label sequences, the raw agreement share and its Wilson 95% interval."""
    a, b = pd.Series(list(a), dtype=object), pd.Series(list(b), dtype=object)
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(str), b[ok].astype(str)
    n = len(a)
    if n == 0:
        return {"n": 0, "kappa": float("nan"), "agreement": float("nan"), "agreement_ci": [float("nan")] * 2}
    po = float((a.to_numpy() == b.to_numpy()).mean())
    cats = sorted(set(a) | set(b))
    pa = a.value_counts(normalize=True).reindex(cats).fillna(0.0).to_numpy()
    pb = b.value_counts(normalize=True).reindex(cats).fillna(0.0).to_numpy()
    pe = float(pa @ pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": n, "kappa": kappa, "agreement": po, "agreement_ci": wilson(int(round(po * n)), n)}


def wilson(k: int, n: int, level: float = LEVEL) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [c - h, c + h]


def fisher_ci(r: float, n: int, level: float = LEVEL) -> list[float]:
    """Fisher-z interval of a correlation (n pairs)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return [float("nan"), float("nan")]
    z = math.atanh(r)
    h = float(stats.norm.ppf(0.5 + level / 2)) / math.sqrt(n - 3)
    return [math.tanh(z - h), math.tanh(z + h)]
