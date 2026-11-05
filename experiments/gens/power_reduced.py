"""Sample sizes and MDEs of the reduced pre-unlock generations (2026-09-28 budget re-plan) next to the design's
(design_v1 §13.2, §13.6-§13.9; design_v2_delta IF-2, LA-2). Planning arithmetic only, from the design's own power
model (`experiments/plan/plan_v1.py: power`) and the W_pre-slice power inputs (`results/pre/power_inputs.json`);
nothing is estimated on W_clean.

    .venv/bin/python experiments/gens/power_reduced.py --out results/gens/power_reduced.json

Conventions (plan_v1.py): z2 = 1.96 + 0.8416 (two-sided 5%, power 80%); z_holm = 2.128 + 0.8416 (one-sided alpha/3,
power 80%). Weekly IC sd for a cross-section of n names: sigma_n^2 = sigma_U100^2 - 1/99 + 1/(n - 1) (the part of the
U100 sd above pure rank noise is kept; the rank-noise part scales with the names), under two sigma_U100: the design's
planning value 0.12 and the W_pre-slice measurement (PRE, 58 weeks).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Z2, Z_HOLM, Z95 = 1.96 + 0.8416, 2.128 + 0.8416, 1.96


def sigma_n(sigma_u100: float, n: int) -> float:
    return math.sqrt(max(sigma_u100**2 - 1 / 99, 0.0) + 1 / (n - 1))


def ic_block(n_names: int, weeks: int, sig_u: float) -> dict:
    s = sigma_n(sig_u, n_names)
    out = {"sigma_ic": round(s, 4), "ci95_halfwidth": round(Z95 * s / math.sqrt(weeks), 4),
           "mde_single": round(Z2 * s / math.sqrt(weeks), 4)}
    for rho in (0.2, 0.5):
        out[f"mde_paired_rho{rho}"] = round(Z2 * s * math.sqrt(2 * (1 - rho)) / math.sqrt(weeks), 4)
    return out


def auc_se_null(n_pos: float, n_neg: float) -> float:
    """Hanley-McNeil SE of an AUC at 0.5 (Mann-Whitney null variance)."""
    return math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))


def prop_mde(p0: float, n: float, z: float = Z2) -> float:
    return z * math.sqrt(p0 * (1 - p0) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/gens/power_reduced.json"))
    a = ap.parse_args()
    pi = json.loads((ROOT / "results/pre/power_inputs.json").read_text())
    sig = {"plan_0.12": 0.12, "wpre_measured": round(float(pi["sd_ic_primary"]), 4)}
    res: dict = {"conventions": {"z2": Z2, "z_holm": Z_HOLM, "sigma_u100": sig,
                                 "source": "plan_v1.py power(); results/pre/power_inputs.json (W_pre slice)"}}

    # TAL / MCOT (S5 estimation; TAL also C1-secondary and T4 sample pools)
    for run in ("TAL", "MCOT"):
        res[run] = {
            "design": {"units": "W_clean x S50", "weeks": 115, "names": 50, "stock_weeks": 115 * 50,
                       **{k: ic_block(50, 115, v) for k, v in sig.items()},
                       "ir_mde": round(Z2 / math.sqrt(115 / 52), 2)},
            "reduced": {"units": "last 60 W_clean decisions x S30", "weeks": 60, "names": 30, "stock_weeks": 60 * 30,
                        **{k: ic_block(30, 60, v) for k, v in sig.items()},
                        "ir_mde": round(Z2 / math.sqrt(60 / 52), 2)},
        }
    # C1-secondary: TA-lite under attack (100 sw x 4 families), paired binary success, plan_v1's C1 formula
    res["TAL"]["c1_secondary_ta_lite_under_attack"] = {
        "pairs": 400, "note": "unchanged n (100 stock-weeks x A1, A2, A3, A6), drawn from the S30 x last-60 pool",
        **{f"mde_pid{p}": round(Z_HOLM * math.sqrt(2 * p / 400), 3) for p in (0.2, 0.4)}}
    res["TAL"]["t4_pool"] = {"design": "500 of 5,750 S50 stock-weeks", "reduced": "500 of 1,800 S30 stock-weeks"}

    # VERB (X5, VC-11, VC-13 descriptive rows): 6,000 -> 2,000 N1 items
    verb = {}
    for n in (6000, 2000):
        verb[str(n)] = {"items": n, "items_per_week_115": round(n / 115, 1),
                        "spearman_pooled_mde": round(Z2 / math.sqrt(n - 3), 4),
                        "weekly_spearman_mean_mde": round(Z2 * (1 / math.sqrt(max(n / 115 - 3, 1))) / math.sqrt(115), 4),
                        **{f"auc_mde_prev{p}": round(Z2 * auc_se_null(n * p, n * (1 - p)), 4) for p in (0.1, 0.3)},
                        "ece_sd_ratio_vs_6000": round(math.sqrt(6000 / n), 2)}
    res["VERB"] = {"design": verb["6000"], "reduced": verb["2000"],
                   "note": "IF-2 aligned the 6,000-item sample to contain NG-C2's 4,000 items; 2,000 items can only be "
                           "a seeded subset of that set, so H2 per implementation compares emu / VERB / NG on those "
                           "2,000"}

    # PROBE: D1 monthly probe 2024-01..2025-06 (18 months) and D2 re-identification
    d1 = {}
    for names, wk in ((20, 4), (10, 2)):
        n = names * wk
        se = auc_se_null(n / 2, n / 2)
        d1[f"{names}x{wk}"] = {"per_month": n, "total": n * 18, "month_auc_se_null": round(se, 4),
                               "month_auc_mde": round(Z2 * se, 3), "month_ci95_halfwidth": round(Z95 * se, 3),
                               "week_blocks_per_month": wk,
                               "three_month_auc_mde": round(Z2 * auc_se_null(1.5 * n, 1.5 * n), 3)}
    res["PROBE"] = {"D1": {"design": d1["20x4"], "reduced": d1["10x2"],
                           "note": "the window-start rule (design_v1 §2.5) needs the July-2024 AUC CI; with 2 weeks per "
                                   "month a week-block bootstrap has 2 blocks per month, so the reduced rule should "
                                   "pool 3-month windows centered on each month (to be stated in the pre-registration)"}}
    d2 = {}
    for n in (2000, 600):
        d2[str(n)] = {"states": n, "reid_company_mde_vs_chance_0.2": round(prop_mde(0.2, n), 4),
                      "reid_year_mde_vs_chance_0.125": round(prop_mde(0.125, n), 4),
                      "per_cell_6": round(n / 6), "reid_company_mde_per_cell": round(prop_mde(0.2, n / 6), 4)}
    res["PROBE"]["D2"] = {"design": d2["2000"], "reduced": d2["600"],
                          "cells": "3 blinding levels x 2 periods (W_pre slice, W_clean)"}

    # FID: V1 joint vs single; V2 re-ask floor
    def v2(nreq: int, rep: int) -> dict:
        df = nreq * (rep - 1)
        pairs = nreq * rep * (rep - 1) / 2
        return {"requests": nreq, "repeats": rep, "answers": nreq * rep, "sd_df_per_question": df,
                "sd_rel_se": round(1 / math.sqrt(2 * df), 4),
                "flip_rate_se_at_0.05": round(math.sqrt(0.05 * 0.95 / nreq), 4), "pairs": int(pairs)}

    res["FID"] = {"V1": {"design": {"stock_weeks": 300, "mean_abs_dp_se_ratio": 1.0},
                         "reduced": {"stock_weeks": 150, "mean_abs_dp_se_ratio": round(math.sqrt(300 / 150), 2)}},
                  "V2": {"design": v2(200, 5), "reduced": v2(100, 3)},
                  "note": "an intervention's flip rate is compared with the floor: with 300 intervention answers and "
                          "the floor at 5%, the MDE above the floor is "
                          f"{round(Z2 * math.sqrt(0.05 * 0.95 * (1 / 300 + 1 / 100)), 4)} (reduced) vs "
                          f"{round(Z2 * math.sqrt(0.05 * 0.95 * (1 / 300 + 1 / 200)), 4)} (design), per-request floor"}

    # MEM (X2: M-work off = T re-ask, M-epi off = N3 re-ask), Holm within X2; IC over the re-asked names
    mem = {}
    for label, n in (("design_U100", 100), ("reduced_S30", 30)):
        row = {}
        for k, v in sig.items():
            s = sigma_n(v, n)
            row[k] = {"sigma_ic": round(s, 4), **{f"mde_holm_rho{r}": round(Z_HOLM * s * math.sqrt(2 * (1 - r))
                                                                             / math.sqrt(115), 4) for r in (0.9, 0.95)}}
        mem[label] = {"names": n, "weeks": 115, "stock_weeks": 115 * n, **row}
    res["MEM"] = {"design": mem["design_U100"], "reduced": mem["reduced_S30"],
                  "note": "rho = correlation of the weekly ICs with and without the store; an ablation changes one "
                          "program's features, so rho is high (0.9-0.95 assumed; PRE's V-codeonly contrast measured "
                          f"{round(pi['contrasts']['V-codeonly']['rho'], 3)} on the S30 slice)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
