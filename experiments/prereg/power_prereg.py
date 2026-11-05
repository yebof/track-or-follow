"""Sample sizes, MDEs and cost projections for the pre-registration draft (docs/design/preregistration_v1.md), after
the 2026-09-28 budget re-plan (user: total paid LLM budget $250).

    nice -n 15 .venv/bin/python experiments/prereg/power_prereg.py --out results/prereg/power_prereg.json

Planning arithmetic only. Conventions are the design's (experiments/plan/plan_v1.py: power): z2 = 1.96 + 0.8416
(two-sided 5%, power 80%); z_holm = 2.128 + 0.8416 (one-sided at alpha/3, the Holm worst case, power 80%). H1 uses
plan_v1's paired-binary formula with a week-cluster design effect of 2: MDE = z * sqrt(2 * pi_d / n_pairs).

Inputs, none of them an outcome on W_clean:
- the design's planning values (results/plan/plan_v1.json) and the W_pre-slice power inputs (results/pre/power_inputs.json);
- the reduced-generation MDEs already computed from the same model (results/gens/power_reduced.json);
- H2 sample sizes: MAIN's C2-eligible item rows by week (Layer-A answers only: status.kept and top event_type !=
  stock_activity_or_opinion, the rule of experiments/ng/items500.py), and their distinct (focal, d0) label units, d0
  from `created_at` and the session calendar (the `date` column of the ETF bar file; no price is read);
- per-request costs measured in the spend ledger (read-only) for the cost projections of the post-unlock runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ng"))

Z2, Z_HOLM, Z95 = 1.96 + 0.8416, 2.128 + 0.8416, 1.96
W_CLEAN = ("2024-07-05", "2026-09-11")
W_RELEASE = ("2025-04-18", "2026-09-11")
EXCLUDED_TYPE = "stock_activity_or_opinion"


def z_one_sided(alpha: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha)


def h1_block(n_pairs: int, z: float = Z_HOLM) -> dict:
    return {f"mde_pid{p}": round(z * math.sqrt(2 * p / n_pairs), 4) for p in (0.1, 0.2, 0.3, 0.4)}


def sessions() -> pd.DatetimeIndex:
    d = pd.read_parquet(ROOT / "data/raw/bars/etf_daily_2018_2026.parquet", columns=["date"])["date"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(d.unique())))


def event_session(created_at: str, ses: pd.DatetimeIndex) -> pd.Timestamp | None:
    """d0 (design_v1 §13.1 C2): the session of `created_at` if before 16:00 ET, otherwise the next session."""
    local = pd.Timestamp(created_at).tz_convert("America/New_York")
    day = pd.Timestamp(local.date())
    k = ses.searchsorted(day, side="left" if local.hour < 16 else "right")
    return ses[k] if k < len(ses) else None


def c2_frame() -> pd.DataFrame:
    """MAIN's C2-eligible (item, focal) rows in W_clean from emu's Layer-A answers (no outcome)."""
    from items500 import main_item_rows, top_label  # the NG sample's frame rule (experiments/ng/items500.py)
    rows = main_item_rows()
    out = []
    for uid, r in rows.items():
        top = top_label(r.get("event_type"))
        if r["status"]["kept"] and top is not None and top != EXCLUDED_TYPE:
            out.append({"t": r["t"], "symbol": r["symbol"], "eid": r["eid"], "created_at": r["created_at"]})
    return pd.DataFrame(out)


def c2_mde(n_t: np.ndarray, between: float, z: float = Z_HOLM) -> float:
    """Weekly Spearman averaged over weeks: per-week null variance 1/(n_t - 1) plus a between-week variance."""
    n_t = n_t[n_t >= 4]
    se = math.sqrt(float(np.mean(1.0 / (n_t - 1))) + between**2) / math.sqrt(len(n_t))
    return round(z * se, 4)


def h3_block(sigma: float, weeks: int) -> dict:
    return {f"mde_rho{r}": round(Z_HOLM * sigma * math.sqrt(2 * (1 - r)) / math.sqrt(weeks), 4)
            for r in (0.5, 0.8, 0.9, 0.95)}


def ledger_costs() -> dict:
    con = sqlite3.connect(f"file:{ROOT / 'data/cache/spend.sqlite'}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("select tag, usd from spend", con)
    finally:
        con.close()
    g = df.groupby("tag")["usd"].agg(["size", "sum"])
    return {t: {"rows": int(r["size"]), "usd": float(r["sum"]), "usd_per_request": float(r["sum"] / r["size"])}
            for t, r in g.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/prereg/power_prereg.json"))
    a = ap.parse_args()
    plan = json.loads((ROOT / "results/plan/plan_v1.json").read_text())
    pin = json.loads((ROOT / "results/pre/power_inputs.json").read_text())
    red = json.loads((ROOT / "results/gens/power_reduced.json").read_text())
    sig_pre = round(float(pin["sd_ic_primary"]), 4)
    ses = sessions()
    wc = [d for d in ses if pd.Timestamp(W_CLEAN[0]) <= d <= pd.Timestamp(W_CLEAN[1])]
    iso = pd.Series(wc).dt.isocalendar()
    n_clean = int(pd.Series(wc).groupby([iso.year.values, iso.week.values]).ngroups)
    learned = n_clean - 2  # the first decision with a matured label week is the third (design_v1 §2.5)
    res: dict = {"conventions": {"z2": Z2, "z_holm": Z_HOLM, "c1_design_effect": 2,
                                 "source": "experiments/plan/plan_v1.py power(); results/pre/power_inputs.json"},
                 "windows": {"W_clean_decisions": n_clean, "W_clean_learned_decisions": learned}}

    # ---------------------------------------------------------------- H1 (C1): 200 promotion units x A1-A7
    m_sec = 7  # C1-secondary contrasts registered in the draft (H-guard, H-spot, nodef, R-med, R-vote, LM, TA-lite)
    z_sec = z_one_sided(0.05 / m_sec) + 0.8416
    res["H1"] = {
        "design": {"units": 400, "families": 7, "pairs": 2800, **h1_block(2800),
                   "plan_v1_json": {k: v for k, v in plan["power"].items() if k.startswith("C1")}},
        "registered": {"units": 200, "families": 7, "pairs": 1400, **h1_block(1400)},
        "registered_if_A7_units_all_lost": {"pairs": 1200, **h1_block(1200),
                                            "note": "A7 counts only units still eligible at t+2 (design_v1 §13.1)"},
        "per_family_95ci_halfwidth_n200": {f"pid{p}": round(Z95 * math.sqrt(2 * p / 200), 4) for p in (0.1, 0.2, 0.4)},
        "c1_secondary": {"contrasts": m_sec, "z_holm_within_family": round(z_sec, 4),
                         "pairs_same_units": 1400, **h1_block(1400, z_sec),
                         "ta_lite_pairs": 400, "ta_lite": h1_block(400, z_sec)},
    }

    # ---------------------------------------------------------------- H2 (C2) on MAIN
    fr = c2_frame()
    fr["d0"] = [event_session(c, ses) for c in fr["created_at"]]
    n_units = fr.groupby("t").size()
    n_label = fr.drop_duplicates(["t", "symbol", "d0"]).groupby("t").size()
    rel = n_units.index >= W_RELEASE[0]
    res["H2"] = {
        "design": {k: v for k, v in plan["power"].items() if k.startswith("C2")} | {
            "assumed_items_per_week": "about 260 (plan_v1.py)"},
        "main_c2_frame": {"rows": int(len(fr)), "weeks": int(n_units.size),
                          "units_per_week": {"min": int(n_units.min()), "median": float(n_units.median()),
                                             "max": int(n_units.max())},
                          "distinct_focal_d0_per_week": {"min": int(n_label.min()), "median": float(n_label.median()),
                                                         "max": int(n_label.max())},
                          "distinct_focal_d0_total": int(n_label.sum())},
        "registered_W_clean": {f"mde_between_sd{b}_{lab}": c2_mde(v.to_numpy(float), b)
                               for b in (0.0, 0.05) for lab, v in (("units", n_units), ("focal_d0", n_label))},
        "W_release": {f"mde_between_sd{b}_{lab}": c2_mde(v[rel].to_numpy(float), b)
                      for b in (0.0, 0.05) for lab, v in (("units", n_units), ("focal_d0", n_label))},
        "note": "per-week null variance 1/(n_t - 1); items sharing a (focal, d0) share their AR label, so the "
                "focal_d0 column is the conservative effective size. Counts only: no AR was computed.",
    }

    # ---------------------------------------------------------------- H3 (C3) on MAIN
    res["H3"] = {
        "design": {k: v for k, v in plan["power"].items() if k.startswith("C3")} | {"sigma": 0.12, "weeks": 115},
        "registered_W_clean": {"weeks": learned, "sigma_0.12": h3_block(0.12, learned),
                               f"sigma_{sig_pre}": h3_block(sig_pre, learned)},
        "W_release": {"weeks": 73, "sigma_0.12": h3_block(0.12, 73), f"sigma_{sig_pre}": h3_block(sig_pre, 73)},
        "rho_note": "PRE's S30 slice gives rho 0.989 for primary vs V-codeonly, inflated because 70 of 100 names have "
                    "no S1 judgment there (pre_report R-2); rho on U100 is unknown before the unlock",
    }

    # ---------------------------------------------------------------- descriptive (PD1 etc.)
    years = learned / 52
    sys.path.insert(0, str(ROOT / "src"))
    from jevtrader.pipeline.variants import registered_names
    n_trials = len(registered_names())
    res["descriptive"] = {
        "single_system_IC_95ci_halfwidth": {f"sigma_{s}": round(Z95 * s / math.sqrt(learned), 4) for s in (0.12, sig_pre)},
        "single_system_IC_mde": {f"sigma_{s}": round(Z2 * s / math.sqrt(learned), 4) for s in (0.12, sig_pre)},
        "portfolio_IR_mde": round(Z2 / math.sqrt(learned / 52), 2),
        # bailey2014pseudo Theorem 3.1 (Eq. 3.2, p.11): MinBTL < 2 ln N / E[max_N]^2 (years); solved for E[max_N]
        "minbtl": {"years": round(years, 3), "trial_series_full_sweep": n_trials,
                   "expected_max_sharpe_of_N_null_trials": {str(n): round(math.sqrt(2 * math.log(n) / years), 2)
                                                             for n in (2, 10, n_trials)},
                   "max_independent_trials_for_E_max_1": round(math.exp(years / 2), 2),
                   "note": "approximation of bailey2014pseudo Eq. 3.2 (p.11), a necessary, non-sufficient condition"},
    }

    # ---------------------------------------------------------------- reduced baselines (quoted + NG)
    res["baselines_reduced"] = {k: red[k] for k in ("TAL", "MCOT", "VERB", "PROBE", "FID", "MEM") if k in red}
    n_ng = 500
    res["baselines_reduced"]["NG_500"] = {
        "units": n_ng, "weeks": 111,
        "pooled_spearman_mde": round(Z2 / math.sqrt(n_ng - 3), 4),
        "pooled_spearman_95ci_halfwidth_fisher_z": round(Z95 / math.sqrt(n_ng - 3), 4),
        "agreement_share_95ci_halfwidth_at_0.5": round(Z95 * math.sqrt(0.25 / n_ng), 4),
        "note": "descriptive per-implementation H2 row (VC-13, IF-4); weekly means are not estimable at ~4.5 units "
                "per week, so the pooled Spearman with a week-block bootstrap is registered for NG and VERB",
    }

    # ---------------------------------------------------------------- cost projections (planning; ledger unit costs)
    lc = ledger_costs()
    per = {k: lc[k]["usd_per_request"] for k in ("MAIN/N0", "MAIN/N1", "MAIN/N2", "MAIN/N3", "MAIN/T", "DEV2/H",
                                                   "DEV2/H-spot", "DEV2/H-guard", "DEV2/HS") if k in lc}
    main_led = sum(v["usd"] for k, v in lc.items() if k.startswith("MAIN/"))
    main_news = sum(lc[k]["usd"] for k in ("MAIN/N0", "MAIN/N1", "MAIN/N2", "MAIN/N3"))
    n1_per_item = 4.51  # MAIN: N1 requests per asked item (main_report §5.3)
    typed_item = per["MAIN/N0"] + n1_per_item * per["MAIN/N1"] + per["MAIN/N2"] + per["MAIN/N3"]
    nodef_item = n1_per_item * per["MAIN/N1"] + per["MAIN/N3"]
    hguard_rate = lc["DEV2/H-guard"]["rows"] / 400.0  # requests per stock-week in DEV #2 (400 stock-weeks)
    in_band = 30 * n_clean
    unit_weeks = 200 * (6 + 3)  # A1-A6 one week each, A7 three weeks
    proj = {
        "unit_costs_usd": {**{k: round(v, 7) for k, v in per.items()}, "typed_attacked_item": round(typed_item, 6),
                           "nodef_attacked_item": round(nodef_item, 6),
                           "MAIN_all_in_per_stock_week": round(main_led / 11600, 6),
                           "MAIN_news_per_stock_week": round(main_news / 11600, 6),
                           "TAL_per_stock_week_measured": 0.003736, "MCOT_per_stock_week_dry_run": 0.00239},
        "H_clean_U100_design": {"requests": 100 * n_clean, "usd": round(100 * n_clean * per["DEV2/H"], 2)},
        "H_clean_in_band": {"requests": in_band, "usd": round(in_band * per["DEV2/H"], 2), "sub_cap": None},
        "HSPOT_clean_in_band": {"requests": in_band, "usd": round(in_band * per["DEV2/H-spot"], 2)},
        "HGUARD_clean_in_band": {"requests": round(in_band * hguard_rate), "usd": round(in_band * hguard_rate * per["DEV2/H-guard"], 2)},
        "NODEF_clean_unit_stock_weeks": {"stock_weeks": 200, "usd": round(200 * (main_news / 11600 - lc["MAIN/N0"]["usd"] / 11600), 2),
                                         "note": "the 200 promotion units' week-t stock-weeks, typed news cost minus N0"},
        "RT_S_stages": {
            "1_H1_typed_plus_H": round(unit_weeks * (typed_item + per["DEV2/H"]), 2),
            "2_H_guard_plus_H_spot": round(unit_weeks * (per["DEV2/H-guard"] + per["DEV2/H-spot"]), 2),
            "3_TA_lite_under_attack_100x4": round(400 * 0.003736 * 5 / 8, 2),
            "4_nodef_attacked": round(unit_weeks * nodef_item, 2),
            "5_demotion_RB2_100_held_x_A8_1_A8_2": round(100 * 2 * (2 * typed_item + per["DEV2/H"]), 2),
            "unit_weeks_promotion": unit_weeks,
            "note": "attacked-item costs at MAIN's measured per-request costs; unchanged items are cache hits "
                    "(redteam/run.py); TA-lite re-runs the news analyst, bull, bear, trader and risk (5 of 8 calls)",
        },
    }
    caps = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())["ledger"]["sub_caps_usd"]
    for k, cap in (("H_clean_in_band", "H"), ("HSPOT_clean_in_band", "HSPOT"), ("HGUARD_clean_in_band", "HGUARD"),
                   ("NODEF_clean_unit_stock_weeks", "NODEF")):
        proj[k]["sub_cap"] = caps.get(cap)
        proj[k]["fits"] = proj[k]["usd"] <= caps.get(cap, 0)
    st = proj["RT_S_stages"]
    cum, fits = 0.0, {}
    for k in ("1_H1_typed_plus_H", "2_H_guard_plus_H_spot", "3_TA_lite_under_attack_100x4", "4_nodef_attacked",
              "5_demotion_RB2_100_held_x_A8_1_A8_2"):
        cum += st[k]
        fits[k] = {"cumulative_usd": round(cum, 2), "within_RT_S_sub_cap": cum <= caps["RT-S"]}
    st["cumulative_vs_sub_cap"] = fits
    st["sub_cap"] = caps["RT-S"]
    wc_dates = sorted(pd.Timestamp(x) for x in pd.Series(wc).groupby([iso.year.values, iso.week.values]).max())
    proj["MCOT_reduced"] = {"weeks": 34, "first_week": str(wc_dates[-34].date()), "last_week": str(wc_dates[-1].date()),
                            "stock_weeks": 34 * 30, "usd": round(34 * 30 * 0.00239, 2), "sub_cap": caps.get("MCOT"),
                            "ci95_halfwidth_mean_ic": {f"sigma_{s}": round(Z95 * math.sqrt(max(s**2 - 1 / 99, 0) + 1 / 29)
                                                                            / math.sqrt(34), 4) for s in (0.12, sig_pre)}}
    proj["LIVE"] = {"per_week_usd_at_MAIN_rate": round(main_led / 116, 3),
                    "weeks": ["2026-09-25 (pseudo-forward)", "2026-10-02", "2026-10-09", "2026-10-16"],
                    "usd": round(4 * main_led / 116, 2), "sub_cap": caps.get("LIVE")}
    # NG (typed-local, $0): worker seconds from ng_report §4 (medians of the first 20 items: N0 1.77 s, single-question
    # N1 1.76 s) and the running generation's measured seconds per full item (progress.json)
    ngp = json.loads((ROOT / "results/judgments/ng_typed-local_wclean_items500/progress.json").read_text())
    s_item = float(ngp.get("s_per_item_session") or 22.3)
    proj["NG_cpu"] = {"s_per_full_item_measured": round(s_item, 2),
                      "items500_hours": round(500 * s_item / 3600, 2),
                      "X6_NG_2000_items_hours": round(2000 * (1.77 + 2 * 1.76) / 3600, 2),
                      "RT_A_NG_30_items_x_64_queries_hours": round(30 * 64 * s_item / 3600, 2),
                      "NG_D1_360_requests_hours": round(360 * 1.76 / 3600, 2),
                      "budget_cpu_hours_IF1": 36,
                      "note": "hours of one 6-thread worker; IF-1's budget is 36 CPU-hours of that worker"}
    res["cost_projection"] = proj
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("H1", "H3")}, indent=1)[:3000])
    print(json.dumps(res["H2"], indent=1)[:2000])
    print(json.dumps(res["cost_projection"], indent=1)[:3000])


if __name__ == "__main__":
    main()
