"""Learned code-only baselines B8 (logistic, L2) and B9 (LightGBM) on the trusted code features (design_v1
section 13.3).

Protocols (section 2.5):
- P-clean: cold start at the window start; learn only from labels of decisions inside the window. Until the first
  label matures (decision 3), the book is EW-U100.
- P-full: expanding window of labels from 2020.
Labels are T1 (`learn.labels.LabelStore`) and are read only through `mature(c_t)`: the label of decision s is usable
at decision t iff it was realized (open of e_{s+1}) before c_t. Features are rank-Gaussian within U_s, computed at
each decision from data up to it, so every training row is an as-of row. The book is top-20 equal weight by the
predicted probability (the D-sel-EW rule without text, hence without a band).
Hyper-parameters are fixed a priori: logistic C = 1 (sklearn's default L2; no intercept, T1 has base rate 1/2 each
week); LightGBM as V-GBM (section 6.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from jevtrader.baselines.classic import top_k_scores
from jevtrader.data.news import decision_cutoff

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200, colsample_bytree=0.8,
                   subsample=0.8, subsample_freq=1, verbose=-1)
LOGIT_C = 1.0


def rank_gaussian(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank-Gaussian x = Phi^-1((rank - 1/2) / n) per column over non-missing values; missing -> 0."""
    r = df.rank(method="average")
    n = df.notna().sum()
    z = norm.ppf((r - 0.5) / n.where(n > 0))
    return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0.0)


def standardized_panel(raw: dict[pd.Timestamp, pd.DataFrame], features: tuple[str, ...]) -> pd.DataFrame:
    """Long frame (t, symbol) x features, each week standardized within its own cross-section."""
    frames = []
    for t, df in sorted(raw.items()):
        x = rank_gaussian(df.reindex(columns=list(features)).astype(float))
        frames.append(x.assign(t=pd.Timestamp(t), symbol=x.index))
    return pd.concat(frames, ignore_index=True).set_index(["t", "symbol"]).sort_index()


def _model(kind: str, seed: int):
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=LOGIT_C, fit_intercept=False, max_iter=1000)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    raise ValueError(kind)


@dataclass
class LearnedBaseline:
    kind: str  # "logit" (B8) | "lgbm" (B9)
    protocol: str  # "P-clean" | "P-full"
    train_start: pd.Timestamp  # only labels of decisions on or after this date are learned from
    features: tuple[str, ...]
    k: int = 20
    seed: int = 0
    log: list = field(default_factory=list)

    def run(self, X: pd.DataFrame, labels, decisions: list[pd.Timestamp],
            universe: dict[pd.Timestamp, list[str]]) -> tuple[dict, pd.DataFrame]:
        """Returns (targets {t: weights}, predictions (t, symbol, p, cold))."""
        lab = labels.table[["t", "symbol", "y", "available_at"]]
        lab = lab[lab.t >= pd.Timestamp(self.train_start)]
        targets, preds = {}, []
        for t in decisions:
            t = pd.Timestamp(t)
            members = list(universe[t])
            c = decision_cutoff(t)
            tr = lab[(lab.available_at < c) & lab.y.notna()]  # == LabelStore.mature(c), restricted to the protocol
            tr = tr.join(X, on=["t", "symbol"], how="inner")
            xt = X.loc[t].reindex(members).fillna(0.0) if t in X.index.get_level_values(0) else None
            if tr.empty or tr.y.nunique() < 2 or xt is None:
                targets[t] = {s: 1.0 / len(members) for s in members}  # cold start: EW-U100
                preds += [(t, s, np.nan, True) for s in members]
                self.log.append((t, 0, 0))
                continue
            m = _model(self.kind, self.seed).fit(tr[list(self.features)].to_numpy(), tr.y.to_numpy().astype(int))
            p = pd.Series(m.predict_proba(xt[list(self.features)].to_numpy())[:, 1], index=xt.index)
            targets[t] = top_k_scores(p, self.k)
            preds += [(t, s, float(v), False) for s, v in p.items()]
            self.log.append((t, len(tr), tr.t.nunique()))
        return targets, pd.DataFrame(preds, columns=["t", "symbol", "p", "cold"])


def code_feature_panel(panels, fund_panel, sector_map, dates: list[pd.Timestamp],
                       members: dict[pd.Timestamp, list[str]]) -> dict[pd.Timestamp, pd.DataFrame]:
    """Raw (unstandardized) trusted code features of U_t at every decision date t, each from data up to t:
    technical from adjusted bars (sector ETF by the trailing-correlation map), fundamentals from filings with
    filed < t and raw closes. `panels`: data.prices.PricePanels; `fund_panel`: data.fundamentals.FundamentalsPanel;
    `sector_map`: features.technical.SectorMap."""
    from jevtrader.features import technical as T

    out = {}
    for t in dates:
        m = members[t]
        snap = T.technical_snapshot(panels.adjusted, t, m, sector_etf=sector_map.asof(t, m), raw=panels.raw)
        out[t] = T.code_features(snap).join(fund_panel.features(t, m), how="outer").reindex(m)
    return out
