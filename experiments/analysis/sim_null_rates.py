#!/usr/bin/env python3
"""Null rejection rates of the two pre-registered tests the pre-commit audit (2026-09-28) questioned, by simulation
(development tool; synthetic data only, no file of the study is read).

    .venv/bin/python experiments/analysis/sim_null_rates.py            # -> results/analysis/simulations/null_rates.json

**H1 (prereg §3.1).** 200 promotion units spread over 113 learned weeks, families A1-A7 (A7 measured at t + 2, the
others at the attack week t), two readers. Each unit has a baseline propensity a ~ N(logit .2, 1) and a unit-level
typed-minus-holistic propensity d ~ N(0, SD_D); success_typed ~ Bern(sigmoid(a + d/2)) and success_H ~
Bern(sigmoid(a - d/2)) for every family, so E[ASR_typed - ASR_H] = 0 exactly (the null). The one-sided p of the
registered paired bootstrap (eval.metrics.asr_cluster_bootstrap, p = mean(boot - est <= est)) is computed with two
cluster definitions: the measurement week eval_t (the pre-audit code) and the unit's attack week t_attack (the
registered one). Also the case d = 0 (no unit effect). Rejection rates at .05 and .10.

**D1 (prereg §7.2).** A pooled 3-month window: 6 decision weeks x the same 10 names, P(beat) independent of the label
y. Two nulls: iid (p ~ U(0, 1), y ~ Bern(.5)), and persistent names (p = sigmoid(a_name + 0.3 e), y = 1[0.8 b_name +
e' > 0], a and b independent). The share of null windows the rule would call "different from 0.5": the pre-audit
percentile week-block bootstrap interval excluding 0.5 (stats.week_block_bootstrap_xy with AUC), and the registered
two-sided name-permutation test at p < .05 (stats.name_permutation_auc). A center window that "differs" moves the
confirmatory start, so this is the false-move rate.

Seeds are fixed (numpy default_rng); the output records every setting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

OUT = Path("results/analysis/simulations/null_rates.json")
H1_UNITS, H1_WEEKS, H1_FAMILIES = 200, 113, ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
SD_D = 1.5
D1_WEEKS, D1_NAMES = 6, 10


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def h1_null(reps: int, n_boot: int, sd_d: float, seed: int) -> dict:
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    rng = np.random.default_rng(seed)
    p = {"eval_t": [], "t_attack": []}
    for r in range(reps):
        t = rng.integers(0, H1_WEEKS - 2, H1_UNITS)
        a = rng.normal(np.log(0.2 / 0.8), 1.0, H1_UNITS)
        d = rng.normal(0.0, sd_d, H1_UNITS) if sd_d > 0 else np.zeros(H1_UNITS)
        nf = len(H1_FAMILIES)
        ta = np.repeat(t, nf)
        ev = ta + np.tile([2 if f == "A7" else 0 for f in H1_FAMILIES], H1_UNITS)
        pa, pb = np.repeat(_sig(a + d / 2), nf), np.repeat(_sig(a - d / 2), nf)
        sa, sb = (rng.random(len(pa)) < pa).astype(float), (rng.random(len(pb)) < pb).astype(float)
        for name, cl in (("eval_t", ev), ("t_attack", ta)):
            p[name].append(asr_cluster_bootstrap(sa, cl, sb, n_boot=n_boot, seed=r)["p_one_sided"])
    return {k: {"reject_05": float(np.mean(np.asarray(v) <= 0.05)), "reject_10": float(np.mean(np.asarray(v) <= 0.10)),
                "reps": reps} for k, v in p.items()}


def d1_null(reps: int, n_boot: int, scenario: str, seed: int) -> dict:
    from jevtrader.eval.metrics import auc

    rng = np.random.default_rng(seed)
    boot = perm = 0
    for r in range(reps):
        an, bn = rng.normal(0, 1, D1_NAMES), rng.normal(0, 1, D1_NAMES)
        rows = []
        for w in range(D1_WEEKS):
            for k in range(D1_NAMES):
                if scenario == "iid":
                    pv, yv = rng.random(), float(rng.random() < 0.5)
                else:
                    pv, yv = _sig(an[k] + 0.3 * rng.normal()), float(0.8 * bn[k] + rng.normal() > 0)
                rows.append({"t": w, "pos": k, "p": pv, "y": yv})
        f = pd.DataFrame(rows)
        ci = S.week_block_bootstrap_xy(f, "t", "p", "y", auc, n_boot=n_boot, seed=r)["ci"]
        boot += int(np.isfinite(ci[0]) and not (ci[0] <= 0.5 <= ci[1]))
        perm += int(S.name_permutation_auc(f, "t", "pos", "p", "y", n_perm=n_boot, seed=r)["p_two_sided"] < 0.05)
    return {"bootstrap_interval_excludes_0.5": boot / reps, "name_permutation_p_below_0.05": perm / reps,
            "reps": reps, "n_boot_or_perm": n_boot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h1-reps", type=int, default=4000)
    ap.add_argument("--d1-reps", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=500, help="H1 bootstrap size per replication")
    ap.add_argument("--d1-boot", type=int, default=1000, help="D1 bootstrap / permutation size per replication")
    ap.add_argument("--out", default=str(C.ROOT / OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    res = {"h1": {"setting": {"units": H1_UNITS, "weeks": H1_WEEKS, "families": list(H1_FAMILIES), "a7_offset": 2,
                              "sd_d": SD_D, "n_boot": a.n_boot, "seed": 0},
                  "unit_effect": h1_null(a.h1_reps, a.n_boot, SD_D, 0),
                  "no_unit_effect": h1_null(max(a.h1_reps // 2, 1), a.n_boot, 0.0, 1)},
           "d1": {"setting": {"weeks": D1_WEEKS, "names": D1_NAMES, "alpha": 0.05, "seed": 2},
                  "iid": d1_null(a.d1_reps, a.d1_boot, "iid", 2),
                  "persistent_names": d1_null(a.d1_reps, a.d1_boot, "persistent", 3)}}
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    C.write_result(out, res, script=__file__, window="synthetic", guard=C.Guard(C.ROOT), argv=argv)
    print(f"H1 reject at .05: eval_t {res['h1']['unit_effect']['eval_t']['reject_05']:.4f}, t_attack "
          f"{res['h1']['unit_effect']['t_attack']['reject_05']:.4f}; D1 false moves (persistent): bootstrap "
          f"{res['d1']['persistent_names']['bootstrap_interval_excludes_0.5']:.3f}, permutation "
          f"{res['d1']['persistent_names']['name_permutation_p_below_0.05']:.3f} -> {C.rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
