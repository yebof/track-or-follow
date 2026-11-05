"""The allocator (design §6.2-6.4): trusted learner, untrusted learner with the logged as-of offset, text cap, pooled
probability and exact per-agent attribution.

    L^T = sum_{j in T} beta_j x~_j                       beta fitted on trusted features and price labels only
    L^W = sum_{j in W} gamma_j x~_j                      gamma fitted with offset o = logged as-of L^T (§6.7)
    Lbar^W = sign(L^W) min(|L^W|, kappa),  kappa = 2 max(sd_i L^T, 0.01)
    L = L^T + Lbar^W,  P = sigmoid(L)
    c_a = sum_{j in a, T} beta_j x~_j + rho sum_{j in a, W} gamma_j x~_j,  rho = Lbar^W / L^W,  sum_a c_a = L

beta, R^T and kappa depend on trusted inputs only, so poisoned text in any week (current or earlier) cannot move
them; that is what the band guard's influence bound rests on (§3.4). Learners read only the ledger: standardized
values and L^T as logged at their decision time, joined to labels released by `LabelStore.mature`.

Exactness of the attribution (card logit ledger; dev1_report §11.3 / P2-4; docs/impl/dev2_ops.md §4). L^T, L^W and
every c_a are fixed-order compensated (Neumaier) sums of the logged question contributions c_j, computed with
elementwise IEEE-754 operations, so they are reproducible bit for bit on any machine and library version (no reduction
order inside numpy or pandas is involved) and each is within about one rounding of the exact sum of its c_j. Of the
card's identities, L = L^T + Lbar^W holds bit for bit by construction (it is how L is computed). The agent line,
sum_a c_a = L, cannot also hold bit for bit in floating point: the card sums the rounded c_a (Python's compensated
`sum`), while L rounds L^T + Lbar^W; forcing equality would mean nudging some c_a away from the rounded sum of its
own c_j, which only moves the rounding to the question level. Its residual is rounding only, bounded a priori by
`attribution_bound` (valid for a compensated or a plain left-to-right sum over the agents): |fl(sum_a c_a) - L| <= u ((n_agents + 5) sum_j |c_j| + |Lbar^W| + |L|)
with u = 2^-53, which is below 1e-12 for any sum_j |c_j| < ~500 (a logit that large would mean P = 1 - 1e-200).
DEV #1 measured at most 1.67e-16 (L^T + L^W + cap line = L: 0 on every card).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from jevtrader.decide.select import rank_desc
from jevtrader.learn.features import CATALOG, TRUSTED_V1, UNTRUSTED_V1
from jevtrader.learn.labels import LabelStore, _as_utc
from jevtrader.learn.online import OnlineLogit, decay_weights

WEEK_S = 7 * 86400.0
UNIT_ROUNDOFF = 2.0 ** -53  # IEEE-754 double, round to nearest


def fixed_sum(cols: Sequence[np.ndarray], n: int | None = None) -> np.ndarray:
    """Row-wise sum of equally long float arrays in the given order, with Neumaier compensation.

    Elementwise IEEE operations only, so the result does not depend on numpy's or pandas' internal reduction order
    (pairwise, SIMD width, memory layout): identical bits on every machine. The error is at most about u |sum| +
    2 k u^2 sum |x| for k terms (u = 2^-53), i.e. the sum is correctly rounded up to a u^2 term."""
    if not cols:
        return np.zeros(0 if n is None else n)
    s = np.array(cols[0], dtype=float, copy=True)
    c = np.zeros_like(s)
    for x in cols[1:]:
        x = np.asarray(x, dtype=float)
        t = s + x
        c += np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        s = t
    return s + c


def agents_card_sum(agents: pd.DataFrame) -> np.ndarray:
    """sum_a c_a exactly as a decision card adds it: decide/cards.py `check_card` takes Python's built-in `sum` over
    the agents in column order, which for floats is Neumaier-compensated since Python 3.12 (the environment runs
    3.12), the same algorithm as `fixed_sum`, so the two agree bit for bit (tests/test_pipeline_attribution_bound.py)."""
    return fixed_sum([agents[a].to_numpy(dtype=float) for a in agents.columns], len(agents))


def attribution_bound(scores: Scores) -> pd.Series:
    """A priori bound on |sum_a c_a - L| per name, the card's agent line (module doc): u ((n_agents + 5) sum_j |c_j| +
    |Lbar^W| + |L|), with a 1e-6 relative margin for evaluating the bound itself in floating point.

    Derivation (u = 2^-53; A = sum_j |c_j| over the logged question contributions): each compensated sum (c_a, L^T,
    L^W) is within 1.01 u of its exact value relative to its terms' absolute sum; with the cap, rho = fl(Lbar^W / L^W)
    and c_j = fl(rho gamma_j x_j) add u |Lbar^W| + 2.02 u A_W; L = fl(L^T + Lbar^W) adds u |L|; summing the n agents
    adds at most gamma_{n-1} sum_a |c_a| <= 1.01 (n - 1) u A even left to right (the card's compensated sum adds about
    u |L|). In total <= u ((n + 4.04) A + |Lbar^W| + |L|)."""
    n_a = scores.agents.shape[1]
    A = scores.questions.abs().sum(axis=1).to_numpy(dtype=float)
    f = scores.frame
    b = UNIT_ROUNDOFF * ((n_a + 5) * A + np.abs(f["L_W_bar"].to_numpy(dtype=float)) + np.abs(f["L"].to_numpy(dtype=float)))
    return pd.Series(b * (1 + 1e-6), index=scores.frame.index)


def attribution_residual(scores: Scores) -> pd.Series:
    """|sum_a c_a - L| per name, with the sum taken as the decision card takes it."""
    return pd.Series(np.abs(agents_card_sum(scores.agents) - scores.frame["L"].to_numpy(dtype=float)),
                     index=scores.frame.index)


def trusted_not_T(features: Sequence[str], catalog: Mapping = CATALOG) -> list[str]:
    """Catalog features in `features` whose channel is not T (unknown names are left to `specs_for`)."""
    return [f for f in features if f in catalog and catalog[f].channel != "T"]


@dataclass(frozen=True)
class AllocatorConfig:
    trusted: tuple[str, ...] = TRUSTED_V1
    untrusted: tuple[str, ...] = UNTRUSTED_V1
    prior_sd_beta: float = 0.04
    prior_sd_gamma: float = 0.04
    half_life_weeks: float | None = 52.0
    cap_mult: float = 2.0  # math.inf = V-nocap
    cap_floor: float = 0.01
    namespace: str = ""  # backend / pinned model / bank hash / prompt version / blinding

    def __post_init__(self):
        # P0 rests on L^T reading T-channel features only. A config with an untrusted side claims the trust split, so
        # a W-channel catalog feature on its trusted side is refused (from_yaml, run_layer_b --trusted, variants). A
        # config without an untrusted side claims no split (V-joint: one learner over everything, no cap) and is
        # exempt; eval.bound.bound_check reports such a trusted set as not all-T and does not pass it.
        if self.untrusted:
            bad = trusted_not_T(self.trusted)
            if bad:
                raise ValueError(f"W-channel features on the trusted side of a split config: {bad} (catalog channel "
                                 "!= 'T'); move them to `untrusted`, or drop the untrusted side (no split, as V-joint)")

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.trusted) + tuple(self.untrusted)

    def agent_of(self, catalog: Mapping = CATALOG) -> dict[str, str]:
        return {f: catalog[f].agent for f in self.features}


class Ledger:
    """As-of log of what second-level fits may consume: standardized features and the trusted logit L^T exactly as
    computed at each decision, with the calibration snapshot id. Rows are appended once and never rewritten."""

    def __init__(self, features: Sequence[str]):
        self.features = list(features)
        self._weeks: list[pd.DataFrame] = []
        self._frame: pd.DataFrame | None = None
        self.last_t: pd.Timestamp | None = None

    def append(self, t, x: pd.DataFrame, L_T: pd.Series, snapshot_id: str) -> None:
        t = pd.Timestamp(t)
        if self.last_t is not None and t <= self.last_t:
            raise ValueError("ledger is append-only in decision order")
        f = x.reindex(columns=self.features).astype(float).copy()
        f.insert(0, "symbol", f.index.astype(str))
        f.insert(0, "t", t)
        f["L_T_asof"] = L_T.reindex(x.index).to_numpy(dtype=float)
        f["calib_snapshot_id"] = snapshot_id
        self._weeks.append(f.reset_index(drop=True))
        self._frame, self.last_t = None, t

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            cols = ["t", "symbol", *self.features, "L_T_asof", "calib_snapshot_id"]
            self._frame = pd.concat(self._weeks, ignore_index=True) if self._weeks else pd.DataFrame(columns=cols)
        return self._frame


@dataclass(frozen=True)
class CalibSnapshot:
    snapshot_id: str
    cutoff_utc: pd.Timestamp
    trusted: tuple[str, ...]
    untrusted: tuple[str, ...]
    beta: np.ndarray
    gamma: np.ndarray
    cov_beta: np.ndarray
    cov_gamma: np.ndarray
    n_train: int
    n_weeks: int
    sum_w: float
    namespace: str = ""

    @property
    def cold(self) -> bool:
        """No matured label yet: the coefficients are the prior (0) and every book holds EW-U100 (§2.5)."""
        return self.n_train == 0

    def to_json(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id, "cutoff_utc": self.cutoff_utc.isoformat(), "namespace": self.namespace,
            "beta": dict(zip(self.trusted, map(float, self.beta))), "gamma": dict(zip(self.untrusted, map(float, self.gamma))),
            "se_beta": dict(zip(self.trusted, map(float, np.sqrt(np.diag(self.cov_beta))))),
            "se_gamma": dict(zip(self.untrusted, map(float, np.sqrt(np.diag(self.cov_gamma))))),
            "cov_beta": self.cov_beta.tolist(), "cov_gamma": self.cov_gamma.tolist(),
            "n_train": self.n_train, "n_weeks": self.n_weeks, "sum_w": self.sum_w,
        }


@dataclass(frozen=True)
class Scores:
    frame: pd.DataFrame  # symbol -> L_T, L_W, L_W_bar, rho, cap_line, L, P, R_T, R
    agents: pd.DataFrame  # symbol -> c_a per agent (sum over agents == L)
    questions: pd.DataFrame  # symbol -> c_j per feature (beta_j x~_j, or rho gamma_j x~_j)
    x: pd.DataFrame  # the standardized features scored
    kappa: float
    s_T: float
    snapshot_id: str
    cold: bool


class Allocator:
    def __init__(self, cfg: AllocatorConfig, labels: LabelStore, catalog: Mapping = CATALOG):
        self.cfg, self.labels = cfg, labels
        self.agent = cfg.agent_of(catalog)
        self.agents = list(dict.fromkeys(self.agent[f] for f in cfg.features))

    def training_rows(self, ledger: Ledger, cutoff_utc) -> pd.DataFrame:
        """Ledger rows joined to labels matured before the cutoff, with their decay weights, in (t, symbol) order."""
        c = _as_utc(cutoff_utc)
        lab = self.labels.mature(c)[["t", "symbol", "y", "available_at"]]
        led = ledger.frame()
        if led.empty or lab.empty:
            return pd.DataFrame(columns=[*led.columns, "y", "available_at", "w"])
        rows = led.merge(lab, on=["t", "symbol"], how="inner", validate="one_to_one")
        rows = rows.sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)
        age = (c - rows["available_at"]).dt.total_seconds().to_numpy() / WEEK_S
        rows["w"] = decay_weights(age, self.cfg.half_life_weeks)
        return rows

    def fit(self, ledger: Ledger, cutoff_utc) -> CalibSnapshot:
        cfg = self.cfg
        rows = self.training_rows(ledger, cutoff_utc)
        y, w = rows["y"].to_numpy(dtype=float), rows["w"].to_numpy(dtype=float)
        tr = OnlineLogit(cfg.prior_sd_beta).fit(rows[list(cfg.trusted)].to_numpy(dtype=float), y, w)
        un = OnlineLogit(cfg.prior_sd_gamma).fit(
            rows[list(cfg.untrusted)].to_numpy(dtype=float), y, w, offset=rows["L_T_asof"].to_numpy(dtype=float)
        )
        c = _as_utc(cutoff_utc)
        n_weeks = int(rows["t"].nunique()) if len(rows) else 0
        payload = json.dumps(
            [cfg.namespace, c.isoformat(), cfg.trusted, cfg.untrusted, tr.coef_.tobytes().hex(), un.coef_.tobytes().hex(), len(rows)]
        )
        return CalibSnapshot(
            snapshot_id=hashlib.sha256(payload.encode()).hexdigest()[:16], cutoff_utc=c,
            trusted=tuple(cfg.trusted), untrusted=tuple(cfg.untrusted), beta=tr.coef_.copy(), gamma=un.coef_.copy(),
            cov_beta=tr.cov_, cov_gamma=un.cov_, n_train=len(rows), n_weeks=n_weeks, sum_w=float(w.sum()),
            namespace=cfg.namespace,
        )

    def score(self, feats: pd.DataFrame, snap: CalibSnapshot) -> Scores:
        """Score one week's standardized features (index = U_t)."""
        cfg = self.cfg
        XT = feats.reindex(columns=list(snap.trusted)).fillna(0.0).astype(float)
        XW = feats.reindex(columns=list(snap.untrusted)).fillna(0.0).astype(float)
        qT = XT * snap.beta
        qW = XW * snap.gamma
        n = len(feats.index)
        # fixed-order compensated sums (module doc): reproducible bits, within ~1 rounding of the exact sums
        LT = fixed_sum([qT[c].to_numpy() for c in qT.columns], n)
        LW = fixed_sum([qW[c].to_numpy() for c in qW.columns], n)
        sT = float(np.std(LT, ddof=1)) if len(LT) > 1 else 0.0
        kappa = cfg.cap_mult * max(sT, cfg.cap_floor) if math.isfinite(cfg.cap_mult) else math.inf
        LWb = np.sign(LW) * np.minimum(np.abs(LW), kappa)
        rho = np.divide(LWb, LW, out=np.ones_like(LW), where=LW != 0)
        L = LT + LWb
        idx = feats.index
        qWr = qW.mul(rho, axis=0)
        questions = pd.concat([qT, qWr], axis=1)
        agents = pd.DataFrame({a: fixed_sum([questions[f].to_numpy() for f in questions.columns if self.agent[f] == a], n)
                               for a in self.agents}, index=idx, columns=self.agents)
        frame = pd.DataFrame(
            {"L_T": LT, "L_W": LW, "L_W_bar": LWb, "rho": rho, "cap_line": (rho - 1.0) * LW, "L": L, "P": expit(L)},
            index=idx,
        )
        frame["R_T"] = rank_desc(frame["L_T"])
        frame["R"] = rank_desc(frame["L"])
        return Scores(frame=frame, agents=agents, questions=questions, x=feats.copy(), kappa=float(kappa), s_T=sT,
                      snapshot_id=snap.snapshot_id, cold=snap.cold)
