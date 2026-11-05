"""Feature catalog, the S1 feature map phi (design §5.0) and the point-in-time standardizer (§6.1).

Two families, standardized differently:
- code features: cross-sectional rank-Gaussian within U_t, x~ = Phi^-1((rank - 1/2) / n); missing -> 0.
- S1 features: x~ = clip((x - median over the covered set C) / S, -4, 4) on C and 0 elsewhere. S is the sd of the
  covered, median-demeaned values over the 52 weeks *before* t (the current week only while fewer than 8 weeks of
  history exist), floored at 0.05. A trailing scale keeps an uninformative week from being inflated into a
  full-strength signal, and the sd (not the MAD) survives saturated emulator answers (MAD = 0).

The standardizer is stateful and strictly causal: weeks must be fed in increasing order, and the values it returns
are logged by the pipeline and never recomputed (learners train on the logged as-of values).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRUSTED, UNTRUSTED = "T", "W"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    agent: str
    channel: str  # "T" (trusted: prices, XBRL, macro) or "W" (untrusted Web text)
    kind: str  # "code" (rank-Gaussian) or "s1" (covered-median demeaning + trailing scale)


def _specs(agent: str, channel: str, kind: str, names: str) -> list[FeatureSpec]:
    return [FeatureSpec(n, agent, channel, kind) for n in names.split()]


CATALOG: dict[str, FeatureSpec] = {
    f.name: f
    for f in (
        # trusted set (§6.1): code technical (6), code fundamental (6 + 2 conditional on the M14 tests), S1 (10)
        *_specs("technical", TRUSTED, "code", "mom_12_1 rev_1w rev_1m vol_60d dist_52w_high rel_sector_3m"),
        *_specs("fundamentals", TRUSTED, "code",
                "earnings_yield sales_yield gross_profitability operating_margin rev_growth_ttm share_change_1y "
                "fcf_yield accruals"),
        *_specs("technical", TRUSTED, "s1", "x_trend t_pullback t_capitulation t_breakout t_overextended"),
        *_specs("fundamentals", TRUSTED, "s1", "x_growth x_prof x_strain x_val"),
        *_specs("macro", TRUSTED, "s1", "x_mac"),
        # untrusted set: news (9), sentinel (1), policy (2)
        *_specs("news", UNTRUSTED, "s1", "f_imp f_cont f_rev f_sur f_net f_und f_story f_cov no_news"),
        *_specs("sentinel", UNTRUSTED, "s1", "h"),
        *_specs("policy", UNTRUSTED, "s1", "s_pol s_ind"),
        # readers dropped into the W slot by variants and baselines (§6.3, §6.6, §13.3)
        *_specs("news", UNTRUSTED, "code", "news_count"),  # V-codeonly: ln(1 + #items)
        *_specs("holistic", UNTRUSTED, "s1", "h_outlook"),  # H.outlook, normalized expectation
        # design_v2_delta RB-15 / RB-5: the holistic question over the guarded evidence (H-guard) and over datamarked
        # evidence (H-spot); RB-6: robust-aggregation typed readers, the median over kept clusters of rho nu m d
        # (R-med) and their sign majority (R-vote)
        *_specs("holistic", UNTRUSTED, "s1", "hguard_outlook hspot_outlook"),
        *_specs("news", UNTRUSTED, "s1", "f_med f_vote"),
        *_specs("headline_sentiment", UNTRUSTED, "s1", "hs_tone"),  # HS: mean P(good) - P(bad)
        *_specs("lm_dictionary", UNTRUSTED, "code", "lm_neg"),  # B10
        # forecast-type views, variant V-outlook only (§6.1)
        *_specs("technical", TRUSTED, "s1", "tech_view"),
        *_specs("fundamentals", TRUSTED, "s1", "fund_view"),
        *_specs("news", UNTRUSTED, "s1", "news_view"),
    )
}

# fcf_yield and accruals are in: the M14 tests pass (data.fundamentals.CASHFLOW_READY), so T_code has 14 features,
# the list B8/B9 and prior_check use (configs/learning_v1.yaml holds the frozen copy)
TRUSTED_V1: tuple[str, ...] = tuple(
    "mom_12_1 rev_1w rev_1m vol_60d dist_52w_high rel_sector_3m "
    "earnings_yield sales_yield gross_profitability operating_margin rev_growth_ttm share_change_1y fcf_yield accruals "
    "x_trend t_pullback t_capitulation t_breakout t_overextended x_growth x_prof x_strain x_val x_mac".split()
)
UNTRUSTED_V1: tuple[str, ...] = tuple("f_imp f_cont f_rev f_sur f_net f_und f_story f_cov no_news h s_pol s_ind".split())


def specs_for(names: Sequence[str], catalog: Mapping[str, FeatureSpec] = CATALOG) -> list[FeatureSpec]:
    missing = [n for n in names if n not in catalog]
    if missing:
        raise KeyError(f"unknown features {missing}")
    return [catalog[n] for n in names]


# ------------------------------------------------------------------------------------ feature map phi
T_TREND_VALUES = {"strong_uptrend": 1.0, "uptrend": 0.5, "sideways": 0.0, "downtrend": -0.5, "strong_downtrend": -1.0}
FB_GROWTH_VALUES = {"rapid_growth": 1.0, "steady_growth": 0.5, "recovering": 0.25, "flat": 0.0, "declining": -1.0,
                    "not_available": 0.0}
FB_VALUATION_VALUES = {"looks_cheap": 1.0, "looks_fair": 0.0, "looks_expensive": -1.0, "cannot_tell": 0.0}

# trusted S1 features computed in Layer B from answers.parquet: feature -> (program, qid, kind, value map | levels)
S1_TRUSTED_MAP: dict[str, tuple[str, str, str, object]] = {
    "x_trend": ("T", "trend", "choice", T_TREND_VALUES),
    "t_pullback": ("T", "pullback_in_uptrend", "noul", None),
    "t_capitulation": ("T", "capitulation", "noul", None),
    "t_breakout": ("T", "breakout", "noul", None),
    "t_overextended": ("T", "overextended", "noul", None),
    "x_growth": ("FB", "growth_stage", "choice", FB_GROWTH_VALUES),
    "x_prof": ("FB", "profitability_for_industry", "score", 4),
    "x_strain": ("FB", "financial_strain", "noul", None),
    "x_val": ("FB", "valuation_fit", "choice", FB_VALUATION_VALUES),
    "x_mac": ("M", "industry_tailwind_{industry}", "score", 3),  # MARKET row, the stock's FF12 industry
    "tech_view": ("T", "technical_view", "score", 5),
    "fund_view": ("FB", "fundamental_view", "score", 5),
}


# Layer A's features.parquet names features "<program>.<name>" (programs/runner.py); map them onto the catalog.
LAYER_A_ALIASES = {
    "T.x_trend": "x_trend", "T.pullback_in_uptrend": "t_pullback", "T.capitulation": "t_capitulation",
    "T.breakout": "t_breakout", "T.overextended": "t_overextended", "T.technical_view": "tech_view",
    "FB.growth_stage": "x_growth", "FB.profitability_for_industry": "x_prof", "FB.financial_strain": "x_strain",
    "FB.valuation_fit": "x_val", "FB.fundamental_view": "fund_view", "N3.analyst_view": "news_view",
}


def catalog_name(name: str, catalog: Mapping[str, FeatureSpec] = CATALOG) -> str | None:
    """Catalog name of a Layer-A feature: itself, an explicit alias, or '<program>.<catalog name>'; else None."""
    if name in catalog:
        return name
    if name in LAYER_A_ALIASES:
        return LAYER_A_ALIASES[name]
    tail = name.split(".", 1)[-1]
    return tail if tail in catalog else None


def phi_noul(probs: Mapping[str, float] | None, clip: float = 4.0) -> float | None:
    """x = clip(logit P(yes), -clip, clip); None (abstain) for a degraded answer."""
    if probs is None:
        return None
    p = float(probs.get("yes", 0.0))
    with np.errstate(divide="ignore"):
        z = math.log(p) - math.log1p(-p) if 0.0 < p < 1.0 else (math.inf if p >= 1.0 else -math.inf)
    return float(min(clip, max(-clip, z)))


def phi_choice(probs: Mapping[str, float] | None, values: Mapping[str, float]) -> float | None:
    """Ordered Choice: x = sum_l v(l) p(l); labels without a value count as 0."""
    if probs is None:
        return None
    return float(sum(values.get(k, 0.0) * float(p) for k, p in probs.items()))


def phi_score(probs: Mapping[str, float] | None, n_levels: int) -> float | None:
    """Score with levels 0..L: x = (E[level] - L/2) / (L/2) in [-1, 1]."""
    if probs is None:
        return None
    half = (n_levels - 1) / 2.0
    e = sum(int(k) * float(p) for k, p in probs.items())
    return float((e - half) / half)


def _probs(v) -> dict | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return json.loads(v) if isinstance(v, str) else dict(v)


def features_from_answers(
    answers: pd.DataFrame, industry: Mapping[str, str], features: Sequence[str] | None = None, clip: float = 4.0
) -> pd.DataFrame:
    """Trusted S1 features from a judgment table (`t, symbol, program, qid, probs_json`), long format
    `t, symbol, feature, value, covered`. A degraded answer (probs null) gives covered = False (abstain)."""
    feats = list(features or [f for f in S1_TRUSTED_MAP if f not in ("tech_view", "fund_view")])
    a = answers.copy()
    a["t"] = pd.to_datetime(a["t"])
    a["qid"] = a["qid"].astype(str).str.split(".", n=1).str[-1]  # Layer A writes "<program>.<name>"
    key = {(p, q): g for (p, q), g in a.groupby(["program", "qid"], sort=False)}
    rows = []
    for f in feats:
        prog, qid, kind, param = S1_TRUSTED_MAP[f]
        if f == "x_mac":
            mk = a[(a["program"] == prog) & (a["symbol"] == "MARKET")]
            tail = {(t, q): p for t, q, p in zip(mk["t"], mk["qid"], mk["probs_json"])}
            stocks = a.loc[a["symbol"] != "MARKET", ["t", "symbol"]].drop_duplicates()
            for t, s in zip(stocks["t"], stocks["symbol"]):
                g = industry.get(s)
                q = qid.format(industry=g)
                if g is None or (t, q) not in tail:
                    continue
                x = phi_score(_probs(tail[(t, q)]), param)
                rows.append((t, s, f, np.nan if x is None else x, x is not None))
            continue
        g = key.get((prog, qid))
        if g is None:
            continue
        for t, s, pj in zip(g["t"], g["symbol"], g["probs_json"]):
            pr = _probs(pj)
            x = phi_noul(pr, clip) if kind == "noul" else phi_choice(pr, param) if kind == "choice" else phi_score(pr, param)
            rows.append((t, s, f, np.nan if x is None else x, x is not None))
    return pd.DataFrame(rows, columns=["t", "symbol", "feature", "value", "covered"])


# ------------------------------------------------------------------------------------ standardization
def rank_gaussian(x: pd.Series) -> pd.Series:
    """Phi^-1((rank - 1/2) / n) over the finite values (average ranks for ties); missing -> 0."""
    v = pd.to_numeric(x, errors="coerce").astype(float)
    ok = np.isfinite(v.to_numpy())
    out = pd.Series(0.0, index=x.index)
    n = int(ok.sum())
    if n:
        r = v[ok].rank(method="average").to_numpy()
        out[ok] = norm.ppf((r - 0.5) / n)
    return out


def wide_week(raw_t: pd.DataFrame, universe: Sequence[str], features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, covered) as universe x features from a long `symbol, feature, value[, covered]` slice.
    A name without a row is uncovered; `covered` defaults to value being finite."""
    uni = list(universe)
    r = raw_t[raw_t["feature"].isin(features) & raw_t["symbol"].isin(uni)]
    if r.duplicated(["symbol", "feature"]).any():
        raise ValueError("duplicate (symbol, feature) rows in one week")
    val = r.pivot(index="symbol", columns="feature", values="value").reindex(index=uni, columns=list(features))
    val = val.apply(pd.to_numeric, errors="coerce").astype(float)
    if "covered" in r:
        cov = r.pivot(index="symbol", columns="feature", values="covered").reindex(index=uni, columns=list(features))
        cov = cov.astype("boolean").fillna(False).astype(bool)
    else:
        cov = pd.DataFrame(True, index=uni, columns=list(features))
    cov &= np.isfinite(val.to_numpy())
    return val, cov


@dataclass
class ScaleLog:
    median: float
    scale: float
    n_covered: int
    history_weeks: int


class Standardizer:
    """Point-in-time standardizer (§6.1). `transform` must be called once per decision, in increasing t."""

    def __init__(
        self, specs: Sequence[FeatureSpec], trailing_weeks: int = 52, min_history_weeks: int = 8,
        floor: float = 0.05, clip: float = 4.0,
    ):
        self.specs = list(specs)
        self.trailing = pd.Timedelta(weeks=trailing_weeks)
        self.min_history, self.floor, self.clip = min_history_weeks, floor, clip
        self._hist: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {s.name: [] for s in self.specs if s.kind == "s1"}
        self.last_t: pd.Timestamp | None = None
        self.log: dict[pd.Timestamp, dict[str, ScaleLog]] = {}

    def transform(self, t, raw_t: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
        t = pd.Timestamp(t)
        if self.last_t is not None and t <= self.last_t:
            raise ValueError(f"standardizer is causal: {t.date()} <= last {self.last_t.date()} (history is never re-standardized)")
        names = [s.name for s in self.specs]
        val, cov = wide_week(raw_t, universe, names)
        out = pd.DataFrame(0.0, index=list(universe), columns=names)
        week_log: dict[str, ScaleLog] = {}
        for s in self.specs:
            if s.kind == "code":
                out[s.name] = rank_gaussian(val[s.name].where(cov[s.name]))
                continue
            c = cov[s.name].to_numpy()
            x = val[s.name].to_numpy()[c]
            if not len(x):
                week_log[s.name] = ScaleLog(np.nan, np.nan, 0, 0)
                continue
            med = float(np.median(x))
            dm = x - med
            past = [a for (u, a) in self._hist[s.name] if t - self.trailing <= u < t]
            if len(past) >= self.min_history:
                sd = float(np.std(np.concatenate(past), ddof=1))
            else:
                sd = float(np.std(dm, ddof=1)) if len(dm) > 1 else 0.0
            scale = max(sd if np.isfinite(sd) else 0.0, self.floor)
            col = np.zeros(len(c))
            col[c] = np.clip(dm / scale, -self.clip, self.clip)
            out[s.name] = col
            week_log[s.name] = ScaleLog(med, scale, int(c.sum()), len(past))
            h = self._hist[s.name]
            h.append((t, dm))
            while h and h[0][0] < t - self.trailing:
                h.pop(0)
        self.last_t = t
        self.log[t] = week_log
        return out
