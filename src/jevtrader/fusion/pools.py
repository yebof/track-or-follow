"""Composition variants over the as-of agent heads (design §6.6, design_v2_delta §3.4 MA-1..MA-8; PD2).

Inputs are logged at decision time only (design §6.7): the per-agent head logits l_{a,i,t} of learn/heads.py
(`DecisionLog.heads`), the standardized features of the ledger (V-UAW, V-GBM) and labels released by
`LabelStore.mature(c_t)`. Every learned quantity at decision t (pool weights, reputations, the discount, the best
agent, the CDB lambda and kappa, the GBM) is fitted on rows whose label matured before c_t, so every variant is a
point-in-time replay (tests/test_variant_pools.py). The primary book is never touched: these are registered
variants ($0 replays, MA-12) and the PD2 comparison rows.

| id          | score (log-odds unless stated)                                         | source                   |
|-------------|------------------------------------------------------------------------|--------------------------|
| V-EW-log    | mean_a l_a (equal-weight logarithmic pool)                             | MA-1                     |
| V-EW-lin    | logit(mean_a sigma(l_a)) (linear pool)                                 | MA-1                     |
| V-EW-med    | logit(median_a sigma(l_a))                                             | MA-1                     |
| V-MV        | net votes #(l_a > 0) - #(l_a < 0); ties by V-EW-log (a-priori rule)    | MA-2                     |
| V-REP-best  | l of the agent with the largest Beta posterior mean reputation         | MA-3 (decay 0.9)         |
| V-REP-vote  | reputation-weighted votes, weights max(0, logit(rep_a)); ties V-EW-log | MA-3                     |
| V-SUM       | sum_a l_a (naive independent log-odds sum)                             | MA-4                     |
| V-DISC      | a sum_a l_a, one shared scale a in [0, 1] fitted on matured rows       | MA-4, MA-6               |
| V-UAW       | mean over programs of mean_j s_j x~_j, a-priori signs, no learning     | MA-5 (no probability)    |
| V-2L        | fusion.logpool.LogPool (w >= 0) on the head logits                     | design §6.6; MA-6        |
| V-CDB       | kappa sum_a lambda_a l_a, lambda = (R^-1 t) / t (calibrate-decorrelate)| design §6.6              |
| V-best      | l of the agent with the lowest trailing-26-week log loss               | design §6.6; PD2         |
| V-GBM       | LightGBM on all logged x~, weekly expanding refit                      | design §6.6              |
| V-oracle    | P = the agent probability closest to the realized label (infeasible)   | design §6.6; MA-8        |
V-oracle reads the realized label: it is an evaluation bound only and never makes a book.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from jevtrader.data.news import decision_cutoff
from jevtrader.decide.select import tiebreak_key
from jevtrader.eval import metrics as M
from jevtrader.fusion.logpool import LogPool
from jevtrader.learn.labels import LabelStore
from jevtrader.learn.online import OnlineLogit, decay_weights

WEEK_S = 7 * 86400.0
LEARNED_POOLS = ("V-REP-best", "V-REP-vote", "V-DISC", "V-2L", "V-CDB", "V-best", "V-GBM")
POOL_VARIANTS = ("V-EW-log", "V-EW-lin", "V-EW-med", "V-MV", "V-REP-best", "V-REP-vote", "V-SUM", "V-DISC", "V-UAW",
                 "V-2L", "V-CDB", "V-best", "V-GBM")
EVAL_ONLY = ("V-oracle",)
NO_PROBABILITY = ("V-UAW",)  # no outcome learning, so no probability scale: log loss and Brier are not defined
P_CLIP = 1e-6


@dataclass(frozen=True)
class PoolConfig:
    best_trailing_weeks: int = 26
    rep_prior: tuple[float, float] = (1.0, 1.0)
    rep_decay: float = 0.9
    cdb_n0: float = 2000.0
    cdb_shrink_floor: float = 0.2
    cdb_rho0: float = 0.5
    cdb_kappa_prior: tuple[float, float] = (1.0, 0.25)
    disc_prior: tuple[float, float] = (1.0, 0.5)
    half_life_weeks: float | None = 52.0
    gbm: tuple[tuple[str, float], ...] = (("n_estimators", 300), ("learning_rate", 0.05), ("num_leaves", 15),
                                          ("min_child_samples", 200), ("feature_fraction", 0.8),
                                          ("bagging_fraction", 0.8), ("bagging_freq", 1), ("seed", 0))
    uaw_signs: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_yaml(cls, learning_cfg: Mapping, half_life_weeks: float | None = 52.0) -> PoolConfig:
        v = learning_cfg.get("variants", {})
        rep, cdb, g = v.get("reputation", {}), v.get("cdb", {}), v.get("gbm", {})
        return cls(
            best_trailing_weeks=int(v.get("best_trailing_weeks", 26)),
            rep_prior=tuple(float(x) for x in rep.get("prior", (1.0, 1.0))), rep_decay=float(rep.get("decay", 0.9)),
            cdb_n0=float(cdb.get("n0", 2000)), cdb_shrink_floor=float(cdb.get("shrink_floor", 0.2)),
            cdb_rho0=float(cdb.get("rho0", 0.5)),
            cdb_kappa_prior=tuple(float(x) for x in cdb.get("kappa_prior", (1.0, 0.25))),
            disc_prior=tuple(float(x) for x in v.get("disc_prior", (1.0, 0.5))), half_life_weeks=half_life_weeks,
            gbm=tuple((k, g[k]) for k in sorted(g)) if g else cls.gbm,
            uaw_signs=tuple(sorted((str(k), float(x)) for k, x in v.get("uaw_signs", {}).items())),
        )


# ------------------------------------------------------------------------------------ helpers
def lexi_key(score: pd.Series, tiebreak: pd.Series) -> pd.Series:
    """One float that orders by `score` first and breaks exact ties by `tiebreak` (both larger = better): the
    tiebreak's rank, scaled below half of the smallest gap between distinct scores."""
    s = score.astype(float)
    vals = np.unique(s.to_numpy()[np.isfinite(s.to_numpy())])
    gap = float(np.min(np.diff(vals))) if len(vals) > 1 else 1.0
    pct = tiebreak.rank(method="average", pct=True).fillna(0.5)
    return s + 0.49 * gap * pct


def _decay(rows: pd.DataFrame, cutoff, half_life) -> np.ndarray:
    age = (cutoff - rows["available_at"]).dt.total_seconds().to_numpy() / WEEK_S
    return decay_weights(age, half_life)


def reputations(train: pd.DataFrame, agents: Sequence[str], cfg: PoolConfig) -> pd.Series:
    """Beta posterior mean reliability per agent (MA-3): hit = sign(l_a) agrees with y, an abstaining head (l = 0)
    counts neither; decay `rep_decay` per matured label week, newest weight 1."""
    a0, b0 = cfg.rep_prior
    if train.empty:
        return pd.Series(a0 / (a0 + b0), index=list(agents))
    weeks = sorted(train["t"].unique())
    age = {w: len(weeks) - 1 - k for k, w in enumerate(weeks)}
    dec = train["t"].map(age).map(lambda k: cfg.rep_decay ** k).to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    out = {}
    for a in agents:
        lg = train[a].to_numpy(dtype=float)
        voted = lg != 0
        hit = voted & ((lg > 0) == (y == 1))
        miss = voted & ~hit
        out[a] = (a0 + float((dec * hit).sum())) / (a0 + b0 + float((dec * (hit | miss)).sum()))
    return pd.Series(out)


def best_agent(train: pd.DataFrame, agents: Sequence[str], weeks: int) -> str | None:
    """Agent with the lowest log loss over the last `weeks` matured label weeks (V-best); None before any label."""
    if train.empty:
        return None
    ws = sorted(train["t"].unique())[-weeks:]
    tr = train[train["t"].isin(ws)]
    y = tr["y"].to_numpy(dtype=float)
    ll = {a: M.log_loss(expit(tr[a].to_numpy(dtype=float)), y, M.LOGLOSS_EPS) for a in agents}
    return min(agents, key=lambda a: (ll[a], list(agents).index(a)))


def cdb_lambda(train: pd.DataFrame, agents: Sequence[str], cfg: PoolConfig, cutoff) -> tuple[pd.Series, dict]:
    """Calibrate-then-decorrelate weights (calibrated_learning §6.2): lambda = (R^-1 t) / t with t_a the weighted sd
    of head a's logit (its standardized discriminability), R the decay-weighted head-logit correlation shrunk toward
    the equicorrelation matrix, eta = max(floor, n0 / (n0 + n_eff)), the mean correlation blended toward the
    monoculture prior rho0 while n_eff is small; heads with t < 1e-4 get 1 / (1 + (K - 1) rho); negative weights are
    dropped and lambda re-solved (never a sign flip)."""
    K = len(agents)
    if train.empty or K == 0:
        lam = 1.0 / (1.0 + (K - 1) * cfg.cdb_rho0) if K else 1.0
        return pd.Series(lam, index=list(agents)), {"eta": 1.0, "rho_bar": cfg.cdb_rho0, "n_eff": 0.0}
    w = _decay(train, cutoff, cfg.half_life_weeks)
    X = train[list(agents)].to_numpy(dtype=float)
    ws = w / w.sum()
    mu = ws @ X
    Xc = X - mu
    cov = (Xc * ws[:, None]).T @ Xc
    t = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = cov / np.outer(t, t)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    n_eff = float(w.sum() ** 2 / (w**2).sum())
    off = R[~np.eye(K, dtype=bool)] if K > 1 else np.array([cfg.cdb_rho0])
    rho_s = float(np.mean(off)) if len(off) else cfg.cdb_rho0
    rho_bar = (cfg.cdb_n0 * cfg.cdb_rho0 + n_eff * rho_s) / (cfg.cdb_n0 + n_eff)
    eta = max(cfg.cdb_shrink_floor, cfg.cdb_n0 / (cfg.cdb_n0 + n_eff))
    Req = np.full((K, K), rho_bar)
    np.fill_diagonal(Req, 1.0)
    Rh = (1 - eta) * R + eta * Req
    lam = np.full(K, 1.0 / (1.0 + (K - 1) * rho_bar))
    active = [k for k in range(K) if t[k] >= 1e-4]
    for _ in range(K):
        if not active:
            break
        sub = np.ix_(active, active)
        sol = np.linalg.solve(Rh[sub], t[active]) / t[active]
        if (sol >= 0).all():
            lam[active] = sol
            break
        active = [k for k, s in zip(active, sol) if s >= 0]
    for k in range(K):
        if t[k] >= 1e-4 and k not in active:
            lam[k] = 0.0
    return pd.Series(lam, index=list(agents)), {"eta": eta, "rho_bar": rho_bar, "n_eff": n_eff}


# ------------------------------------------------------------------------------------ the replay
@dataclass
class PoolRun:
    scores: dict[str, pd.DataFrame]  # variant -> t, symbol, score, key, P
    diagnostics: dict = field(default_factory=dict)  # per week: reputations, best agent, lambda, kappa, a, pool scale


def run_pools(heads: pd.DataFrame, labels: LabelStore, cfg: PoolConfig = PoolConfig(),
              ledger: pd.DataFrame | None = None, variants: Sequence[str] = (*POOL_VARIANTS, *EVAL_ONLY)) -> PoolRun:
    """Every composition variant for every decision in `heads` (t, symbol, one column per agent). `labels` must be
    the P-clean store of the evaluated window. `ledger` (Ledger.frame(): logged x~) is needed for V-UAW and V-GBM."""
    agents = [c for c in heads.columns if c not in ("t", "symbol", "head_snapshot_id")]
    H = heads.copy()
    H["t"] = pd.to_datetime(H["t"])
    lab = labels.table[["t", "symbol", "y", "available_at"]]
    J = H.merge(lab, on=["t", "symbol"], how="left")
    feat_cols: list[str] = []
    if ledger is not None:
        feat_cols = [c for c in ledger.columns if c not in ("t", "symbol", "L_T_asof", "calib_snapshot_id")]
        LG = ledger[["t", "symbol", *feat_cols]].copy()
        LG["t"] = pd.to_datetime(LG["t"])
        J = J.merge(LG, on=["t", "symbol"], how="left", suffixes=("", "__x"))
    out: dict[str, list[pd.DataFrame]] = {v: [] for v in variants}
    diag: dict[str, dict] = {}
    signs = dict(cfg.uaw_signs)
    gbm_params = dict(cfg.gbm)
    for t in sorted(J["t"].unique()):
        t = pd.Timestamp(t)
        c = decision_cutoff(t)
        wk = J[J["t"] == t].set_index("symbol")
        train = J[(J["t"] < t) & J["available_at"].notna() & (J["available_at"] < c) & J["y"].notna()]
        L = wk[agents].astype(float)
        ew = L.mean(axis=1)
        pr = expit(L)
        d: dict = {"n_train": int(len(train)), "n_train_weeks": int(train["t"].nunique())}

        def emit(v, score, P, key=None, t=t, wk=wk):
            if v in out:
                out[v].append(pd.DataFrame({"t": t, "symbol": wk.index, "score": score.to_numpy(dtype=float),
                                            "key": (key if key is not None else score).to_numpy(dtype=float),
                                            "P": np.asarray(P, dtype=float)}))

        emit("V-EW-log", ew, expit(ew))
        p_lin = pr.mean(axis=1).clip(P_CLIP, 1 - P_CLIP)
        emit("V-EW-lin", logit(p_lin), p_lin)
        p_med = pr.median(axis=1).clip(P_CLIP, 1 - P_CLIP)
        emit("V-EW-med", logit(p_med), p_med)
        pos, neg = (L > 0).sum(axis=1), (L < 0).sum(axis=1)
        net = (pos - neg).astype(float)
        # a vote has no probability scale; the a-priori rule is the Laplace-smoothed vote share (abstentions half)
        emit("V-MV", net, (pos + 0.5 * (len(agents) - pos - neg) + 1.0) / (len(agents) + 2.0), lexi_key(net, ew))
        s_sum = L.sum(axis=1)
        emit("V-SUM", s_sum, expit(s_sum))
        if {"V-REP-best", "V-REP-vote"} & set(variants):
            rep = reputations(train, agents, cfg)
            d["reputation"] = rep.to_dict()
            ab = max(agents, key=lambda a: (rep[a], -agents.index(a)))
            emit("V-REP-best", L[ab], expit(L[ab]))
            om = np.maximum(0.0, logit(rep.clip(P_CLIP, 1 - P_CLIP))).reindex(agents)
            vote = (np.sign(L) * om).sum(axis=1)
            tot = float(om.sum())
            p_rv = (((L > 0).astype(float) * om + 0.5 * (L == 0).astype(float) * om).sum(axis=1) + 1.0) / (tot + 2.0)
            emit("V-REP-vote", vote, p_rv, lexi_key(vote, ew))
            d["rep_best"] = ab
        if "V-DISC" in variants:
            a_raw = 1.0
            if len(train):
                x = train[agents].sum(axis=1).to_numpy(dtype=float)[:, None]
                a_raw = float(OnlineLogit(cfg.disc_prior[1], np.array([cfg.disc_prior[0]])).fit(
                    x, train["y"].to_numpy(dtype=float), _decay(train, c, cfg.half_life_weeks)).coef_[0])
            a = min(1.0, max(0.0, a_raw))
            d["disc_scale_raw"], d["disc_scale"] = a_raw, a
            emit("V-DISC", a * s_sum, expit(a * s_sum))
        if "V-2L" in variants:
            lp = LogPool(agents, nonneg=True)
            if len(train):
                lp.fit(train[agents].to_numpy(dtype=float), train["y"].to_numpy(dtype=float),
                       sample_weight=_decay(train, c, cfg.half_life_weeks))
            z = pd.Series(lp.logit(L.to_numpy(dtype=float)), index=L.index)
            d["2l_weights"], d["2l_scale"] = dict(zip(agents, map(float, lp.w))), float(lp.w.sum())
            emit("V-2L", z, expit(z))
        if "V-CDB" in variants:
            lam, info = cdb_lambda(train, agents, cfg, c)
            kappa = 1.0
            if len(train):
                xp = (train[agents] * lam).sum(axis=1).to_numpy(dtype=float)[:, None]
                kappa = float(OnlineLogit(cfg.cdb_kappa_prior[1], np.array([cfg.cdb_kappa_prior[0]])).fit(
                    xp, train["y"].to_numpy(dtype=float), _decay(train, c, cfg.half_life_weeks)).coef_[0])
            z = kappa * (L * lam).sum(axis=1)
            d["cdb_lambda"], d["cdb_kappa"], d["cdb"] = lam.to_dict(), kappa, info
            emit("V-CDB", z, expit(z))
        if "V-best" in variants:
            ba = best_agent(train, agents, cfg.best_trailing_weeks)
            d["best_agent"] = ba
            z = L[ba] if ba is not None else pd.Series(0.0, index=L.index)
            emit("V-best", z, expit(z))
        if "V-oracle" in variants:
            y = wk["y"].astype(float)
            p_or = pd.Series(np.where(y == 1, pr.max(axis=1), np.where(y == 0, pr.min(axis=1), np.nan)), index=L.index)
            emit("V-oracle", pd.Series(np.nan, index=L.index), p_or)
        if "V-UAW" in variants and feat_cols:
            prog = {}
            from jevtrader.learn.features import CATALOG

            for f in feat_cols:
                sgn = signs.get(f, 0.0)
                if sgn and f in CATALOG:
                    prog.setdefault(CATALOG[f].agent, []).append(sgn * wk[f].astype(float).fillna(0.0))
            s_u = (pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in prog.values()], axis=1).mean(axis=1)
                   if prog else pd.Series(0.0, index=L.index))
            emit("V-UAW", s_u, np.full(len(s_u), np.nan))
        if "V-GBM" in variants and feat_cols:
            p_g = pd.Series(0.5, index=L.index)
            if len(train) >= 2 * gbm_params.get("min_child_samples", 200) and train["y"].nunique() == 2:
                import lightgbm as lgb

                model = lgb.LGBMClassifier(**{k: v for k, v in gbm_params.items() if k != "seed"},
                                           random_state=int(gbm_params.get("seed", 0)), deterministic=True,
                                           force_row_wise=True, n_jobs=1, verbose=-1)
                model.fit(train[feat_cols].to_numpy(dtype=float), train["y"].to_numpy(dtype=int))
                p_g = pd.Series(model.predict_proba(wk[feat_cols].to_numpy(dtype=float))[:, 1], index=L.index)
            p_g = p_g.clip(P_CLIP, 1 - P_CLIP)
            emit("V-GBM", logit(p_g), p_g)
        diag[t.date().isoformat()] = d
    scores = {v: pd.concat(f, ignore_index=True) for v, f in out.items() if f}
    return PoolRun(scores=scores, diagnostics=diag)


# ------------------------------------------------------------------------------------ VC-1 reference forecasters
def reference_forecasts(ledger: pd.DataFrame, labels: LabelStore, feature: str = "mom_12_1", prior_sd: float = 0.04,
                        half_life_weeks: float | None = 52.0) -> dict[str, pd.DataFrame]:
    """VC-1 reference forecasters for every probability table, in the variants' layout (t, symbol, score, key, P):
    `ref-constant` (0.5), `ref-base-rate` (share of y = 1 among labels matured before c_t) and `ref-momentum` (a
    one-feature walk-forward logistic on the logged rank-Gaussian momentum x~ with the allocator's prior; its key is
    the momentum rank itself). All as of c_t."""
    L = ledger[["t", "symbol", *([feature] if feature in ledger else [])]].copy()
    L["t"] = pd.to_datetime(L["t"])
    lab = labels.table[["t", "symbol", "y", "available_at"]]
    J = L.merge(lab, on=["t", "symbol"], how="left")
    out: dict[str, list] = {"ref-constant": [], "ref-base-rate": [], "ref-momentum": []}
    for t in sorted(J["t"].unique()):
        t = pd.Timestamp(t)
        c = decision_cutoff(t)
        wk = J[J["t"] == t]
        train = J[(J["t"] < t) & J["available_at"].notna() & (J["available_at"] < c) & J["y"].notna()]
        n = len(wk)
        base = float(train["y"].mean()) if len(train) else 0.5
        out["ref-constant"].append(pd.DataFrame({"t": t, "symbol": wk["symbol"].to_numpy(), "score": 0.0, "key": 0.0,
                                                 "P": 0.5}))
        out["ref-base-rate"].append(pd.DataFrame({"t": t, "symbol": wk["symbol"].to_numpy(), "score": 0.0, "key": 0.0,
                                                  "P": np.full(n, base)}))
        if feature in wk:
            b = 0.0
            if len(train):
                b = float(OnlineLogit(prior_sd).fit(train[[feature]].to_numpy(dtype=float), train["y"].to_numpy(dtype=float),
                                                    _decay(train, c, half_life_weeks)).coef_[0])
            x = wk[feature].fillna(0.0).to_numpy(dtype=float)
            out["ref-momentum"].append(pd.DataFrame({"t": t, "symbol": wk["symbol"].to_numpy(), "score": b * x,
                                                     "key": x, "P": expit(b * x)}))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


# ------------------------------------------------------------------------------------ evaluation (PD2, MA-6..MA-10)
def evaluate_scores(scores: pd.DataFrame, labels: LabelStore, start: pd.Timestamp | None = None) -> dict:
    """Weekly rank-IC of the variant's key vs r - m (NW t), and the VC-6 score table of P (resolution first)."""
    s = scores if start is None else scores[scores["t"] >= pd.Timestamp(start)]
    lab = labels.table[["t", "symbol", "r", "m", "y"]]
    j = s.merge(lab, on=["t", "symbol"], how="inner").dropna(subset=["r", "y"])
    out: dict = {"n": int(len(j))}
    if j.empty:
        return out
    if j["key"].notna().any():
        j["rel"] = j["r"] - j["m"]
        ic = M.information_coefficient(j.pivot(index="t", columns="symbol", values="key"),
                                       j.pivot(index="t", columns="symbol", values="rel"))
        out.update({"ic_mean": float(ic.mean()), "ic_t_nw4": M.t_stat_newey_west(ic, 4), "n_weeks": int(ic.notna().sum())})
    if j["P"].notna().any():
        out["scores"] = M.score_table(j["P"].to_numpy(), j["y"].to_numpy())
        out["hit_rate"] = float(((j["P"] > 0.5) == (j["y"] == 1)).mean())
    return out


def oracle_gap_share(loss_pool: float, loss_best: float, loss_oracle: float) -> float:
    """MA-8: share of the best-single-agent-to-oracle log-loss gap the pool captures, (L_best - L_pool) /
    (L_best - L_oracle): 1 = oracle, 0 = best member, negative = worse than the best member. (pappu2026multiagent
    report the relative gap to an at-least-one-correct oracle, p.5.) The oracle picks, per stock-week, the one head
    closest to the realized label; it bounds strategies that choose a head, not pools that combine heads, so a
    combining pool (or V-GBM, which reads the features) can exceed 1."""
    den = loss_best - loss_oracle
    return float((loss_best - loss_pool) / den) if den > 0 else float("nan")


def warmup_curve(scores: pd.DataFrame, labels: LabelStore) -> list[dict]:
    """MA-8 warm-up: cumulative IC and log loss by the number of weeks since the first decision."""
    lab = labels.table[["t", "symbol", "r", "m", "y"]]
    j = scores.merge(lab, on=["t", "symbol"], how="inner").dropna(subset=["r", "y"]).sort_values("t")
    rows, seen = [], []
    for k, (t, g) in enumerate(j.groupby("t", sort=True)):
        seen.append(g)
        a = pd.concat(seen)
        ic = g["key"].corr(g["r"] - g["m"], method="spearman") if g["key"].notna().sum() > 4 else float("nan")
        ll = M.log_loss(a["P"].to_numpy(), a["y"].to_numpy(), M.LOGLOSS_EPS) if a["P"].notna().all() else float("nan")
        rows.append({"week": k + 1, "t": pd.Timestamp(t).date().isoformat(), "ic": float(ic), "cum_log_loss": ll})
    return rows


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def dependence_diagnostics(heads: pd.DataFrame, labels: LabelStore) -> dict:
    """MA-7: dependence between pairs of agent heads. kim2025correlated measure error correlation as the agreement
    rate of two models when both are wrong (multiple choice, p.3-4); with a binary label two wrong heads always
    agree, so the binary analogs are reported: P(b wrong | a wrong) against b's error rate (lift), the phi coefficient
    of the two error indicators and the Jaccard overlap of the error sets (sign of the logit vs y; abstaining heads
    excluded). The within-class correlation of the logits (the average over y = 0, 1; the quantity the CDB pool
    discounts) and the plain logit correlation are reported too."""
    agents = [c for c in heads.columns if c not in ("t", "symbol", "head_snapshot_id")]
    j = heads.merge(labels.table[["t", "symbol", "y"]], on=["t", "symbol"], how="inner").dropna(subset=["y"])
    y = j["y"].to_numpy(dtype=float)
    out = {}
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            la, lb = j[a].to_numpy(dtype=float), j[b].to_numpy(dtype=float)
            both = (la != 0) & (lb != 0)
            ea = ((la > 0) != (y == 1))[both].astype(float)
            eb = ((lb > 0) != (y == 1))[both].astype(float)
            wc = [_corr(la[y == k], lb[y == k]) for k in (0.0, 1.0)]
            pa = float(ea.mean()) if len(ea) else float("nan")
            pb = float(eb.mean()) if len(eb) else float("nan")
            pba = float(eb[ea == 1].mean()) if (ea == 1).any() else float("nan")
            union = float(((ea == 1) | (eb == 1)).sum())
            out[f"{a}|{b}"] = {
                "n": int(both.sum()), "error_rate_a": pa, "error_rate_b": pb, "p_b_wrong_given_a_wrong": pba,
                "error_lift": pba / pb if pb and np.isfinite(pba) else float("nan"), "error_phi": _corr(ea, eb),
                "error_jaccard": float(((ea == 1) & (eb == 1)).sum()) / union if union else float("nan"),
                "within_class_logit_corr": float(np.nanmean(wc)) if np.isfinite(wc).any() else float("nan"),
                "logit_corr": _corr(la, lb),
            }
    return out


def pool_books(scores: pd.DataFrame, K: int, cold: Mapping[pd.Timestamp, bool] | None = None,
               universe: Mapping[pd.Timestamp, Sequence[str]] | None = None) -> dict[pd.Timestamp, dict[str, float]]:
    """D-sel-EW book of a composition variant: 1/K on the top-K names by its key (exact ties by the fixed symbol
    hash). Composition variants have no trust split, so there is no band guard (J = infinity); a week the primary run
    held EW-U100 (cold start) holds EW-U100 here too."""
    books = {}
    for t, g in scores.groupby("t", sort=True):
        t = pd.Timestamp(t)
        if cold is not None and cold.get(t, False) and universe is not None:
            u = list(universe[t])
            books[t] = {s: 1.0 / len(u) for s in u}
            continue
        g = g.assign(h=[tiebreak_key(s) for s in g["symbol"]]).sort_values(["key", "h"], ascending=[False, True],
                                                                            kind="mergesort")
        top = list(g["symbol"].iloc[:K])
        books[t] = {s: 1.0 / len(top) for s in top} if top else {}
    return books


def pool_summary(run: PoolRun, labels: LabelStore, primary: pd.DataFrame | None = None,
                 start: pd.Timestamp | None = None, references: Mapping[str, pd.DataFrame] | None = None) -> dict:
    """PD2 table inputs: every variant's IC and scores, and each probability pool's oracle-gap share against V-best
    and V-oracle (incl. the primary allocator's P when `primary` = predictions with t, symbol, L, P). With
    `references` (reference_forecasts), every probability row also gets skill scores against them (VC-1)."""
    res = {v: evaluate_scores(s, labels, start) for v, s in run.scores.items()}
    if primary is not None:
        pr = primary.rename(columns={"L": "key"})[["t", "symbol", "key", "P"]]
        res["primary"] = evaluate_scores(pr, labels, start)
    for k, s in (references or {}).items():
        res[k] = evaluate_scores(s, labels, start)
    if references:
        lab = labels.table[["t", "symbol", "y"]]
        cands = {**run.scores, **({"primary": primary.rename(columns={"L": "key"})} if primary is not None else {})}
        for v, s in cands.items():
            if v in EVAL_ONLY or v in NO_PROBABILITY or v not in res:
                continue
            sk = {}
            for k, ref in references.items():
                j = s[["t", "symbol", "P"]].merge(ref[["t", "symbol", "P"]].rename(columns={"P": "P_ref"}),
                                                  on=["t", "symbol"]).merge(lab, on=["t", "symbol"]).dropna()
                if start is not None:
                    j = j[j["t"] >= pd.Timestamp(start)]
                sk[k] = M.skill_scores(j["P"].to_numpy(), j["y"].to_numpy(), j["P_ref"].to_numpy())
            res[v]["skill"] = sk
    def ll(v):
        return res.get(v, {}).get("scores", {}).get("log_loss", float("nan"))
    lb, lo = ll("V-best"), ll("V-oracle")
    for v, r in res.items():
        if "scores" in r and v not in EVAL_ONLY and math.isfinite(lb) and math.isfinite(lo):
            r["oracle_gap_share"] = oracle_gap_share(r["scores"]["log_loss"], lb, lo)
    return res
