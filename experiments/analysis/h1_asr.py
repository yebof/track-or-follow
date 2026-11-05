#!/usr/bin/env python3
"""H1 (code C1): per-item guarded decomposition vs holistic typed reading, under the same band and cap (prereg §3.1).

    .venv/bin/python experiments/analysis/h1_asr.py                   # registered: every RT-S arm's scores.parquet
    .venv/bin/python experiments/analysis/h1_asr.py --scores a.parquet b.parquet --window W_pre_slice --out-dir X

**Input.** The RT-S score tables written by `redteam.run.run_static` (`<out_root>/<arm>/scores.parquet`; one row per
(unit, family, reader, scored week) with `success`, `eligible`, the book memberships, ranks, |dL|/2 kappa and RB-4's
signed harm). Default: every results/redteam/rts/*/scores.parquet (the typed arm carries typed, H, H-guard, H-spot,
LM, R-med, R-vote; the nodef arm nodef; the TA-lite stage TA-lite). A (unit, family, reader, week) may appear once.

**Confirmatory test.** Promotion units (R^T in (K, K+J], outside both clean D-sel-EW books), families A1-A7, measured
at eval_t (A7 at t+2; units no longer eligible there were dropped by the planner). success_r = 1[the name enters
D-sel-EW_r(attacked)]; a row counts for a reader only if the name is outside that reader's own clean D-sel-EW
(`eligible`). The statistic is the paired difference ASR_typed - ASR_holistic pooled over families on the (unit,
family) pairs eligible for both readers; the one-sided p (ASR_typed < ASR_holistic) comes from the cluster bootstrap
with B = 10,000 and seed 0 (eval.metrics.asr_cluster_bootstrap: clusters resampled with replacement, the null-centred
bootstrap distribution, p = mean(boot - est <= est)); the 95% CI is the percentile interval. **The cluster is the
unit's attack week t** (`t_attack`; redteam.run.cluster_weeks), which nests all seven family rows of a unit: A7 is
measured at t + 2, so clustering on the measurement week would split a unit's rows over two clusters and make the
one-sided p anti-conservative when outcomes vary by unit (audit 2026-09-28: 0.0725 at a nominal 0.05 with eval_t,
0.0533 with t_attack). Every CI of this script uses the same cluster. Rows excluded for a reader are counted.

**C1-secondary** (Holm within the family; registered m = 7): typed vs each of H-guard, H-spot, nodef, R-med, R-vote,
LM, TA-lite, each on the pairs eligible for both, same bootstrap. Holm runs over all seven registered contrasts: a
contrast that did not run (or has no eligible pair) enters with p = 1 and is listed with its status, so the family is
never corrected less strictly than registered (as confirmatory.py does for H1-H3); m_run is reported alongside.

**Reported, outside every family.** ASR per reader and family and pooled (week-clustered CIs); the bound check on
bound units (must be 0); demotion per stratum (removal or halving of D-sized; exclusion alone); persistence of A7 at
t+3..t+4; stock-level displacement of the combined rank; |dL| / 2 kappa; signed harm of both books (mean, 5% CVaR,
sum, bps of NAV; eval.metrics.harm_summary) with an attack-week-clustered CI of the mean; the BG-6 tightness
witnesses on the replayed units (eval.bound.tightness).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "h1"
TYPED, HOLISTIC = "typed", "H"
SECONDARY = ("H-guard", "H-spot", "nodef", "R-med", "R-vote", "LM", "TA-lite")  # prereg §3.1 C1-secondary, m = 7
CLUSTER = "t_attack"  # the bootstrap cluster: the unit's attack week (redteam.run.cluster_weeks), prereg §3.1
N_BOOT, SEED = 10_000, 0
PROMOTION, BOUND, DEMOTION = "promotion", "bound", "demotion"
DEFAULT_GLOB = "results/redteam/rts/*/scores.parquet"
KEY = ["unit_id", "family", "reader", "offset"]


def score_dates(paths) -> list[pd.Timestamp]:
    """The decision dates stored in score tables (read without the outcome columns), for the lock check."""
    import pyarrow.parquet as pq

    out = set()
    for p in paths:
        cols = [c for c in ("eval_t", "t_attack") if c in pq.read_schema(p).names]
        for c in cols:
            out |= set(pd.to_datetime(pd.read_parquet(p, columns=[c])[c]).dropna().unique())
    return sorted(out)


def load_scores(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["source"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no RT-S score table")
    s = pd.concat(frames, ignore_index=True)
    for c in ("eval_t", "t_attack"):
        if c in s:
            s[c] = pd.to_datetime(s[c])
    if "offset" not in s:
        s["offset"] = 0
    dup = s.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} score rows repeat a (unit, family, reader, week): "
                         f"{s.loc[dup, KEY + ['source']].head(6).to_dict('records')}")
    return s


def _prep(s: pd.DataFrame) -> pd.DataFrame:
    from jevtrader.redteam.run import cluster_weeks

    s = s[s["scored"].astype(bool)].copy()
    s["success"] = s["success"].astype(float)
    s["eligible"] = s["eligible"].astype(bool) if "eligible" in s else True
    s["cluster"] = cluster_weeks(s)  # the unit's attack week (module doc)
    return s


def rate(g: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    e = g[g["eligible"]]
    if not len(e):
        return {"n": 0, "n_excluded": len(g), "asr": None, "ci": None}
    bs = asr_cluster_bootstrap(e["success"], e["cluster"], n_boot=n_boot, seed=seed)
    return {"n": len(e), "n_excluded": int(len(g) - len(e)), "asr": float(e["success"].mean()), "ci": bs["ci"],
            "n_clusters": bs["n_clusters"]}


def paired_contrast(pro: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """ASR_a - ASR_b on the (unit, family) pairs eligible for both, bootstrap clustered on the unit's attack week,
    one-sided p for ASR_a < ASR_b."""
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    x = pro[pro["reader"] == a].set_index(["unit_id", "family"])
    y = pro[pro["reader"] == b].set_index(["unit_id", "family"])
    pairs = x.index.intersection(y.index)
    if not len(pairs):
        return {"status": "not run", "reader_a": a, "reader_b": b, "n_pairs": 0}
    ea = x.loc[pairs, "eligible"].to_numpy(dtype=bool)
    eb = y.loc[pairs, "eligible"].to_numpy(dtype=bool)
    both = pairs[ea & eb]
    out = {"status": "ok", "reader_a": a, "reader_b": b, "n_pairs_scored": len(pairs),
           "n_excluded_a_only": int((~ea & eb).sum()), "n_excluded_b_only": int((ea & ~eb).sum()),
           "n_excluded_both": int((~ea & ~eb).sum()), "n_pairs": len(both), "n_boot": n_boot, "seed": seed}
    if not len(both):
        return {**out, "status": "no eligible pair"}
    r = asr_cluster_bootstrap(x.loc[both, "success"], x.loc[both, "cluster"], y.loc[both, "success"],
                              n_boot=n_boot, seed=seed)
    return {**out, "estimate": r["estimate"], "ci": r["ci"], "p_one_sided": r["p_one_sided"], "asr_a": r["asr_a"],
            "asr_b": r["asr_b"], "n_clusters": r["n_clusters"],
            "discordant": {"a_only": int(((x.loc[both, "success"] > 0) & (y.loc[both, "success"] == 0)).sum()),
                           "b_only": int(((x.loc[both, "success"] == 0) & (y.loc[both, "success"] > 0)).sum())}}


def cluster_mean(x: pd.Series, clusters: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean with an attack-week-clustered percentile CI (the ASR bootstrap applied to a continuous variable)."""
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    ok = np.isfinite(x.to_numpy(dtype=float))
    if not ok.any():
        return {"n": 0, "mean": None, "ci": None}
    bs = asr_cluster_bootstrap(x.to_numpy(dtype=float)[ok], clusters.to_numpy()[ok], n_boot=n_boot, seed=seed)
    return {"n": int(ok.sum()), "mean": bs["estimate"], "ci": bs["ci"]}


def analyse(scores: pd.DataFrame, K: int, J: float, typed: str = TYPED, holistic: str = HOLISTIC,
            secondary=SECONDARY, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    from jevtrader.eval.bound import tightness
    from jevtrader.eval.metrics import harm_summary, holm

    s = _prep(scores)
    s0 = s[s["offset"] == 0]
    pro = s0[s0["kind"] == PROMOTION]
    out: dict = {"n_rows": len(s0), "readers": sorted(s0["reader"].unique())}
    c1 = paired_contrast(pro, typed, holistic, n_boot, seed)
    out["confirmatory"] = {"hypothesis": "H1", "code": "C1", "statistic": "ASR_typed - ASR_holistic (paired, pooled "
                           "over families)", "alternative": "less", **c1}
    sec = []
    for r in secondary:
        sec.append({"reader": r, **paired_contrast(pro, typed, r, n_boot, seed)})
    ran = [x for x in sec if x.get("status") == "ok"]
    # Holm over the registered family: a contrast that did not run enters with p = 1 (module doc)
    ps = [x["p_one_sided"] if x.get("status") == "ok" else 1.0 for x in sec]
    for x, p, adj in zip(sec, ps, holm(ps) if sec else []):
        x["p_holm"] = float(adj)
        if x.get("status") != "ok":
            x["p_entered"] = p
    out["secondary"] = {"m_registered": len(SECONDARY), "m": len(sec), "m_run": len(ran),
                        "rule": "Holm over every registered contrast; a contrast not run enters with p = 1",
                        "contrasts": sec}
    out["per_family"] = [{"kind": k, "reader": r, "family": f, **rate(g, n_boot, seed)}
                         for (k, r, f), g in s0.groupby(["kind", "reader", "family"])]
    out["pooled"] = [{"kind": k, "reader": r, **rate(g, n_boot, seed)} for (k, r), g in s0.groupby(["kind", "reader"])]
    bnd = s0[s0["kind"] == BOUND]
    if len(bnd):
        viol = bnd[bnd["entered_ew"].astype(bool) | bnd["entered_sized"].astype(bool)]
        out["bound_check"] = {"n": len(bnd), "violations": viol[["reader", "unit_id", "family"]].astype(str)
                              .to_dict("records"), "passed": not len(viol)}
    else:
        out["bound_check"] = {"n": 0, "status": "no bound units (n_bound = 0 registered; BG-5 covers the property)"}
    dem = s0[s0["kind"] == DEMOTION]
    out["demotion"] = [{"reader": r, "stratum": st, "family": f, **rate(g, n_boot, seed),
                        "exclusion_asr": float(g.loc[g["eligible"], "removed_sized"].astype(float).mean())
                        if "removed_sized" in g and g["eligible"].any() else None}
                       for (r, st, f), g in dem.groupby(["reader", "stratum", "family"])] if len(dem) else []
    per = s[(s["offset"] > 0) & (s["kind"] == PROMOTION)]
    out["persistence"] = [{"reader": r, "family": f, "offset": int(o), "n": len(g),
                           "share_in_book": float(g["in_ew_att"].astype(float).mean()) if "in_ew_att" in g else None}
                          for (r, f, o), g in per.groupby(["reader", "family", "offset"])] if len(per) else []
    by_reader = {}
    for r, g in pro[pro["eligible"]].groupby("reader"):
        row = {"n": len(g)}
        if {"rank_L_clean", "rank_L_att"} <= set(g.columns):
            row["rank_displacement"] = cluster_mean((g["rank_L_clean"] - g["rank_L_att"]).astype(float), g["cluster"],
                                                    n_boot, seed)
        if "dL_over_2kappa" in g:
            d = g["dL_over_2kappa"].astype(float)
            row["dL_over_2kappa"] = {"mean": float(d.mean()), "max": float(d.max())}
        for book in ("sel_ew", "sized"):
            c = f"harm_{book}_bps"
            if c in g:
                row[f"harm_{book}"] = {**harm_summary(g[c].to_numpy(dtype=float)),
                                       "mean_ci": cluster_mean(g[c].astype(float), g["cluster"], n_boot, seed)["ci"]}
        by_reader[r] = row
    out["by_reader"] = by_reader
    need = {"entered_ew", "entered_sized", "R_T", "protected", "held", "anchor", "v", "v_ref", "w_att", "w_ref",
            "excluded", "entrant", "tau", "phi", "w_cur", "dL_over_2kappa"}
    if need <= set(s0.columns):
        out["tightness"] = tightness(s0.assign(**{c: s0[c].astype(bool) for c in ("entered_ew", "entered_sized",
                                                                                   "protected", "held", "anchor",
                                                                                   "excluded", "entrant")}),
                                     SimpleNamespace(K=K, J=J))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", nargs="*", default=None, help=f"score tables (default: {DEFAULT_GLOB})")
    ap.add_argument("--window", default=C.REGISTERED_WINDOW, help="label of the window the units come from")
    ap.add_argument("--typed", default=TYPED)
    ap.add_argument("--holistic", default=HOLISTIC)
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    paths = [Path(p) for p in a.scores] if a.scores else sorted(root.glob(DEFAULT_GLOB))
    if not paths:
        raise SystemExit(f"H1: no score table ({DEFAULT_GLOB}); RT-S has not run")
    guard.check(score_dates(paths), "H1 scores")  # before any success is read
    scores = load_scores(paths)
    if a.window == C.REGISTERED_WINDOW:  # units drawn before a moved D1 start are outside the confirmatory frame
        d1 = C.d1_start(root)
        if d1["applied"] and (scores["t_attack"] < pd.Timestamp(d1["start"])).any():
            raise SystemExit(f"H1: units before the D1 start {pd.Timestamp(d1['start']).date()}; redraw them")
    cfg = C.load_config(C.SYSTEM_YAML, root)
    K, J = int(cfg["selection"]["K"]), float(cfg["selection"]["J"])
    res = analyse(scores, K, J, a.typed, a.holistic)
    payload = {**res, "settings": {"n_boot": N_BOOT, "seed": SEED, "cluster": CLUSTER, "K": K, "J": J,
                                   "typed": a.typed,
                                   "holistic": a.holistic, "secondary_registered": list(SECONDARY)}}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard,
                   inputs={f"scores {i}": p for i, p in enumerate(paths)}, argv=argv)
    c = res["confirmatory"]
    if c.get("status") == "ok":
        print(f"H1 {a.window}: ASR {c['asr_a']:.3f} vs {c['asr_b']:.3f}, diff {c['estimate']:.4f} CI {c['ci']} "
              f"p {c['p_one_sided']:.4g} ({c['n_pairs']} pairs) -> {C.rel(path, root)}")
    else:
        print(f"H1 {a.window}: {c.get('status')} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
