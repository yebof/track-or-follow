"""Feature-level synthetic attack batches, for development runs and tests of the H1 and Fig. 2 analyses only.

NOT the registered red team (RT-S, redteam.run). No text is planted and no model is called: a batch directory is
written in RT-S's layout (MANIFEST.json with the batch block, features.parquet, no risk table), whose attacked feature
values are set directly, and `redteam.run.score_batch` then replays every unit against every reader's clean run exactly
as it replays real RT-S batches. This exercises unit sampling, batching, the replay scorer and the ASR analyses on
real clean Layer-B logs (e.g. the W_pre slice), with answers known by construction:

- family `PLANT-NONE`: nothing changes, so every reader's attacked decision equals its clean one: success 0 and
  |dL| = 0 for every row;
- family `PLANT-READER`: only the named reader features are pushed, so readers that do not read them (the typed
  programs, when a stand-in reader's features are pushed) have success exactly 0;
- family `PLANT-ALL`: both feature sets are pushed;
- bound units (R^T > K + J) never enter any book under any plant (the band guard, P1).

A pushed feature is set to its clean value plus SHIFT_SD cross-sectional standard deviations of that week's raw
values, in the direction of the reader's week-t coefficient (the attacker knows the week's coefficients, prereg §3.1).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("PLANT-NONE", "PLANT-READER", "PLANT-ALL")
SHIFT_SD = 3.0


def _gamma(log, t) -> dict:
    snap = next((x for x in log.snapshots if x["t"] == pd.Timestamp(t).date().isoformat()), None)
    return dict(snap["gamma"]) if snap else {}


def pushed_values(raw: pd.DataFrame, log, t, symbol: str, features: Sequence[str]) -> dict[str, float]:
    """{feature: attacked raw value} for one stock-week (module doc)."""
    t = pd.Timestamp(t)
    wk = raw[(pd.to_datetime(raw["t"]) == t) & raw["feature"].isin(list(features))]
    g = _gamma(log, t)
    out = {}
    for f in features:
        col = wk[wk["feature"] == f]
        vals = col["value"].astype(float)
        if "covered" in col:
            vals = vals[col["covered"].astype(bool).to_numpy()]
        sd = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 1.0
        sd = sd if np.isfinite(sd) and sd > 0 else 1.0
        own = col.loc[col["symbol"] == symbol, "value"].astype(float)
        base = float(own.iloc[0]) if len(own) and np.isfinite(own.iloc[0]) else float(np.nanmedian(vals)) \
            if vals.notna().any() else 0.0
        sign = np.sign(g.get(f, 0.0)) or 1.0
        out[f] = base + sign * SHIFT_SD * sd
    return out


def write_batches(out_root: Path, units: pd.DataFrame, raw: pd.DataFrame, typed_log, typed_features: Sequence[str],
                  reader_log, reader_features: Sequence[str], families: Sequence[str] = FAMILIES) -> list[Path]:
    """One batch per (family, unit week) in RT-S's layout; returns the batch directories."""
    out_root = Path(out_root)
    dirs = []
    for fam in families:
        for t, g in units.groupby("t", sort=True):
            t = pd.Timestamp(t)
            bid = f"{fam}_{t:%Y%m%d}"
            d = out_root / bid
            d.mkdir(parents=True, exist_ok=True)
            rows, metas = [], []
            for u in g.itertuples(index=False):
                vals: dict[str, float] = {}
                if fam in ("PLANT-ALL",):
                    vals.update(pushed_values(raw, typed_log, t, u.symbol, typed_features))
                if fam in ("PLANT-READER", "PLANT-ALL"):
                    vals.update(pushed_values(raw, reader_log, t, u.symbol, reader_features))
                rows += [{"t": t, "symbol": u.symbol, "feature": f, "value": v} for f, v in vals.items()]
                metas.append({"unit_id": u.unit_id, "kind": u.kind, "symbol": u.symbol,
                              "stratum": getattr(u, "stratum", ""), "R_T": int(u.R_T), "affected": [u.symbol]})
            man = {"batch": {"batch_id": bid, "family": fam, "t": str(t.date()), "eval_t": str(t.date()),
                             "weeks": [str(t.date())], "affected": sorted(set(g["symbol"])), "units": metas},
                   "arm": "synthetic-feature-plants", "note": "feature-level synthetic plants (experiments/analysis/"
                                                             "feature_plants.py); not RT-S"}
            (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
            pd.DataFrame(rows, columns=["t", "symbol", "feature", "value"]).to_parquet(d / "features.parquet",
                                                                                         index=False)
            dirs.append(d)
    return dirs


def score(dirs: Sequence[Path], runs: Mapping[str, tuple], raw: pd.DataFrame, labels, root: Path) -> pd.DataFrame:
    """redteam.run.score_batch over the batches; `runs` maps a reader label to (config, DecisionLog)."""
    from jevtrader.redteam import run as RT

    clean = [RT.CleanRun(k, cfg, log, raw) for k, (cfg, log) in runs.items()]
    return pd.concat([RT.score_batch(d, clean, labels=labels, root=root) for d in dirs], ignore_index=True)
