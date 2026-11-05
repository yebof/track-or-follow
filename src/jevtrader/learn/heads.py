"""Per-program ("agent") logistic heads, fitted and logged as-of at every decision (design §6.6-6.7; design_v2_delta
§3.4: "Agent heads are the as-of per-program logistic heads (V-2L inputs, logged at decision time)").

For each agent a (technical, fundamentals, macro, news, sentinel, policy, ... as in the feature catalog), a MAP
logistic head on a's own standardized features only, with the allocator's skeptical prior and decay:

    l_{a,i,t} = sum_{j in a} b_{a,j} x~_{j,i,t},    b_a = argmin sum_{available_at < c_t} w logloss + ||b||^2 / (2 s^2)

s is the allocator's prior sd of the agent's channel (sigma_beta for trusted agents, sigma_gamma for text agents).
Heads are fitted on the ledger's logged x~ and labels released by `LabelStore.mature(c_t)` only, like the allocator,
and their logits are logged at decision time; the composition variants (fusion/pools.py) read only these logged
logits, never a refit (tests/test_asof_stacking.py, tests/test_variant_pools.py). The heads never feed the primary
book: the band-guarded allocator does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jevtrader.learn.allocator import WEEK_S, AllocatorConfig, Ledger
from jevtrader.learn.features import CATALOG, TRUSTED
from jevtrader.learn.labels import LabelStore, _as_utc
from jevtrader.learn.online import OnlineLogit, decay_weights


@dataclass(frozen=True)
class HeadSnapshot:
    snapshot_id: str
    cutoff_utc: pd.Timestamp
    agents: tuple[str, ...]
    features: dict[str, tuple[str, ...]]
    coef: dict[str, np.ndarray]
    n_train: int

    @property
    def cold(self) -> bool:
        return self.n_train == 0

    def to_json(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "cutoff_utc": self.cutoff_utc.isoformat(), "n_train": self.n_train,
                "coef": {a: dict(zip(self.features[a], map(float, self.coef[a]))) for a in self.agents}}


class AgentHeads:
    def __init__(self, cfg: AllocatorConfig, labels: LabelStore, catalog: Mapping = CATALOG):
        self.cfg, self.labels = cfg, labels
        feats = list(cfg.features)
        self.agents = list(dict.fromkeys(catalog[f].agent for f in feats))
        self.features = {a: tuple(f for f in feats if catalog[f].agent == a) for a in self.agents}
        self.prior = {a: (cfg.prior_sd_beta if all(catalog[f].channel == TRUSTED for f in self.features[a])
                          else cfg.prior_sd_gamma) for a in self.agents}

    def fit(self, ledger: Ledger, cutoff_utc) -> HeadSnapshot:
        c = _as_utc(cutoff_utc)
        lab = self.labels.mature(c)[["t", "symbol", "y", "available_at"]]
        led = ledger.frame()
        rows = (led.merge(lab, on=["t", "symbol"], how="inner", validate="one_to_one")
                .sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)) if len(led) and len(lab) else None
        coef = {}
        n = 0 if rows is None else len(rows)
        if rows is not None and n:
            y = rows["y"].to_numpy(dtype=float)
            w = decay_weights((c - rows["available_at"]).dt.total_seconds().to_numpy() / WEEK_S, self.cfg.half_life_weeks)
        for a in self.agents:
            k = len(self.features[a])
            if not n:
                coef[a] = np.zeros(k)
                continue
            X = rows[list(self.features[a])].to_numpy(dtype=float)
            coef[a] = OnlineLogit(self.prior[a]).fit(X, y, w).coef_.copy()
        payload = json.dumps([self.cfg.namespace, c.isoformat(), {a: coef[a].tobytes().hex() for a in self.agents}, n])
        return HeadSnapshot(hashlib.sha256(payload.encode()).hexdigest()[:16], c, tuple(self.agents),
                            dict(self.features), coef, n)

    def score(self, feats: pd.DataFrame, snap: HeadSnapshot) -> pd.DataFrame:
        """symbol x agent head logits for one week's standardized features."""
        out = pd.DataFrame(index=feats.index)
        for a in snap.agents:
            X = feats.reindex(columns=list(snap.features[a])).fillna(0.0).to_numpy(dtype=float)
            out[a] = X @ snap.coef[a]
        return out
