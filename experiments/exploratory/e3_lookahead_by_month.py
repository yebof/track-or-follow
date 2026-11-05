#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E3: look-ahead by month. The D1 recall probe's AUC per month with permutation
null bands (figure-ready), the registered W_release sensitivities of H2 and H3 next to their pre-release complement,
and both related to the emulator's documented knowledge cutoff (2024-06-01) and the pinned snapshot's release date
(2025-04-14).

    .venv/bin/python experiments/exploratory/e3_lookahead_by_month.py
        # -> results/exploratory/e3_lookahead_by_month.json + results/exploratory/figdata/e3_*.csv

Why exploratory. The D1 rule (prereg §7.2) reads one window (centred on 2024-07) and moved nothing; the monthly windows
are registered outputs (results/analysis/d1.json) and are repeated here, not re-decided. What is new: (i) the
name-permutation NULL BAND of every window (the 2.5-97.5% quantiles of the permuted AUC under the rule's own
permutation scheme, stats.name_permutation_auc, B = 10,000, seed 0; the permuted p-value is re-derived from the same
draws and must equal d1.json's), so a figure can show each AUC against the band a no-recall probe would produce;
(ii) single-month AUCs (20 probes in 2 weeks each; very noisy; with their own null band and a descriptive week-block
bootstrap interval, B = 10,000, seed 0); (iii) the same windows over the W_pre slice's months (2023-04..2024-05, the
development run results/analysis/W_pre_slice/d1.json, reproduced), where the emulator's training data certainly
reaches (before the documented cutoff); (iv) H2 and H3 on the complement of W_release (W_clean decisions before
2025-04-18) and by calendar month, next to the registered W_release sensitivities (h2.json / h3.json, repeated). No
new test decides anything and no p-value is presented as confirmatory: the permutation p-values are the registered
statistic re-derived; the interval of a difference is descriptive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402
import d1_window_start as D1  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

NAME = "e3_lookahead_by_month"
DOCUMENTED_CUTOFF = "2024-06-01"  # docs/pricing/openai_model_gpt-4.1-mini.html ("Jun 01, 2024 knowledge cutoff")
SNAPSHOT_RELEASE = "2025-04-14"  # the pinned snapshot gpt-4.1-mini-2025-04-14 (prereg §1)
FIGDIR = X.OUT_DIR / "figdata"
N_PERM, N_BOOT, SEED = D1.N_PERM, D1.N_BOOT, D1.SEED


def perm_null(g: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """stats.name_permutation_auc's draws (same construction, same RNG sequence), returning the permuted AUCs too."""
    X_ = g[["t", "pos", "p", "y"]].copy()
    weeks = sorted(X_["t"].unique())
    K = int(X_["pos"].max()) + 1
    wi = {w: i for i, w in enumerate(weeks)}
    Xm = np.full((len(weeks), K), np.nan)
    Ym = np.full((len(weeks), K), np.nan)
    for w, k, xv, yv in X_.itertuples(index=False):
        Xm[wi[w], int(k)], Ym[wi[w], int(k)] = float(xv), float(yv)
    yflat = Ym.ravel()
    est = float(S._pairwise_auc(Xm.ravel()[None, :], yflat)[0])
    rng = np.random.default_rng(seed)
    boots, done = [], 0
    while done < n_perm:
        m = min(500, n_perm - done)
        perms = np.argsort(rng.random((m, K)), axis=1)
        Xp = Xm[:, perms].transpose(1, 0, 2).reshape(m, -1)
        boots.append(S._pairwise_auc(Xp, yflat))
        done += m
    b = np.concatenate(boots)
    p = (1 + int(np.sum(np.abs(b - 0.5) >= abs(est - 0.5) - 1e-12))) / (1 + n_perm)
    return {"auc": est, "p_perm": p, "null_band_95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "null_band_90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))], "null_mean": float(b.mean())}


def probe_frame(root: Path, guard, months: list[str], lo) -> pd.DataFrame:
    cal = D1.rule_calendar(root, lo)
    uni = D1._universe(root, cal)
    sched = D1.schedule(cal, uni, months)
    ans = D1.union_answers([root / "results/judgments" / g for g in D1.REGISTERED_PROBES])
    frame = sched.merge(ans[["t", "symbol", "p"]], on=["t", "symbol"], how="left")
    guard.check(frame["t"].unique(), "E3 D1 labels")
    hi = pd.Period(months[-1], "M").end_time.normalize()
    px = C.load_prices(guard, lo, hi)
    dates = sorted(frame["t"].unique())
    labels = C.load_labels(px, dates, {t: uni[t] for t in dates})
    return frame.merge(labels.table[["t", "symbol", "y"]], on=["t", "symbol"], how="left")


def windows(frame: pd.DataFrame, months: list[str], registered: dict | None) -> tuple[list, list, dict]:
    from jevtrader.eval.metrics import auc

    pooled, single, check = [], [], {"max_abs_diff_auc": 0.0, "max_abs_diff_p": 0.0, "n": 0}
    for i, m in enumerate(months):
        pool = months[max(0, i - 1): i + 2]
        g = frame[frame["month"].isin(pool)].dropna(subset=["p", "y"])
        r = perm_null(g)
        reg = (registered or {}).get(m)
        row = {"center": m, "months": pool, "edge": len(pool) < 3, "n": int(len(g)), "weeks": int(g["t"].nunique()),
               "positives": int(g["y"].sum()), **r, "differs_from_0_5": bool(r["p_perm"] < D1.ALPHA)}
        if reg is not None:
            row["ci_week_block_descriptive"] = reg["ci"]
            check["max_abs_diff_auc"] = max(check["max_abs_diff_auc"], abs(reg["auc"] - r["auc"]))
            check["max_abs_diff_p"] = max(check["max_abs_diff_p"], abs(reg["p_perm"] - r["p_perm"]))
            check["n"] += 1
        pooled.append(row)
        g1 = frame[frame["month"] == m].dropna(subset=["p", "y"])
        s1 = {"month": m, "n": int(len(g1)), "weeks": int(g1["t"].nunique()), "positives": int(g1["y"].sum())}
        if len(g1) and 0 < g1["y"].sum() < len(g1) and g1["t"].nunique() >= 2:
            s1.update(perm_null(g1))
            s1["ci_week_block_descriptive"] = S.week_block_bootstrap_xy(g1, "t", "p", "y", auc, n_boot=N_BOOT,
                                                                        seed=SEED)["ci"]
        single.append(s1)
    return pooled, single, check


def weekly_block(series: pd.Series) -> dict:
    t = S.nw_test(series.dropna().to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean": t["mean"], "ci": t["ci"], "se": t["se"], "n_weeks": int(t["n"])}


def diff_desc(a: dict, b: dict) -> dict:
    d = a["mean"] - b["mean"]
    se = float(np.hypot(a["se"], b["se"]))
    return {"diff": d, "ci_descriptive": [d - 1.96 * se, d + 1.96 * se],
            "note": "difference of two NW means with independent-sample se (descriptive; no test)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    guard.check_window("W_clean", "E3")
    d1 = C.read_json(root / "results/analysis/d1.json")
    reg = {w["center"]: w for w in d1["monthly_windows"]}
    months = D1.months_between(*D1.PROBE_MONTHS)
    fr = probe_frame(root, guard, months, pd.Timestamp(f"{D1.PROBE_MONTHS[0]}-01"))
    pooled, single, chk = windows(fr, months, reg)
    # the W_pre slice's months (the development run of the same rule), before the documented cutoff
    pre_path = root / "results/analysis/W_pre_slice/d1.json"
    pre_reg = {w["center"]: w for w in C.read_json(pre_path)["monthly_windows"]} if pre_path.exists() else {}
    w0, w1 = C.resolve_window("W_pre_slice", root)
    pmonths = D1.months_between(D1.month_key(w0), D1.month_key(w1))
    fp = probe_frame(root, guard, pmonths, w0)
    ppooled, psingle, pchk = windows(fp, pmonths, pre_reg)
    for chk_ in (chk, pchk):
        if chk_["n"] and (chk_["max_abs_diff_auc"] > 1e-12 or chk_["max_abs_diff_p"] > 1e-12):
            raise SystemExit(f"E3: D1 windows do not reproduce the recorded AUC / permutation p: {chk_}")

    # H2 / H3: registered W_release sensitivities (repeated) and the pre-release complement (exploratory)
    h2 = C.read_json(root / "results/analysis/h2.json")
    h3 = C.read_json(root / "results/analysis/h3.json")
    rel0 = C.release_start(root)
    h2w = pd.Series({pd.Timestamp(k): v["rho"] for k, v in h2["confirmatory"]["by_week"].items()}, dtype=float)
    h2n = pd.Series({pd.Timestamp(k): v["n"] for k, v in h2["confirmatory"]["by_week"].items()})
    icw = pd.DataFrame({pd.Timestamp(k): v for k, v in h3["ic_by_week"].items()}).T.sort_index()
    dic = (icw["primary"] - icw["codeonly"]).astype(float)
    h2_rel_re = weekly_block(h2w[h2w.index >= rel0])
    h3_rel_re = weekly_block(dic[dic.index >= rel0])
    cons = {"h2_W_release_recomputed_minus_registered": h2_rel_re["mean"] - h2["sensitivities"]["W_release"]["mean"],
            "h3_W_release_recomputed_minus_registered": h3_rel_re["mean"] - h3["sensitivities"]["W_release"]["mean"],
            "h2_full_recomputed_minus_registered": weekly_block(h2w)["mean"] - h2["confirmatory"]["mean"],
            "h3_full_recomputed_minus_registered": weekly_block(dic)["mean"] - h3["confirmatory"]["mean"],
            "d1_registered_windows": chk, "d1_wpre_windows": pchk}
    if max(abs(v) for k, v in cons.items() if isinstance(v, float)) > 1e-12:
        raise SystemExit(f"E3: weekly series do not reproduce the registered means: {cons}")
    flagged = [w["center"] for w in pooled if w["differs_from_0_5"]]
    flagged_months = sorted({m for w in pooled if w["differs_from_0_5"] for m in w["months"]})
    mk = lambda idx: pd.Index([D1.month_key(t) for t in idx])  # noqa: E731
    periods = {
        "W_clean_full": (h2w.index, dic.index),
        "pre_release_complement": (h2w.index[h2w.index < rel0], dic.index[dic.index < rel0]),
        "W_release": (h2w.index[h2w.index >= rel0], dic.index[dic.index >= rel0]),
        "d1_flagged_months": (h2w.index[mk(h2w.index).isin(flagged_months)], dic.index[mk(dic.index).isin(
            flagged_months)]),
        "not_d1_flagged": (h2w.index[~mk(h2w.index).isin(flagged_months)], dic.index[~mk(dic.index).isin(
            flagged_months)]),
    }
    by_period = {k: {"h2_weekly_spearman": weekly_block(h2w.loc[i2]), "h3_delta_ic": weekly_block(dic.loc[i3]),
                     "h3_ic_primary": weekly_block(icw.loc[i3, "primary"].astype(float)),
                     "h3_ic_codeonly": weekly_block(icw.loc[i3, "codeonly"].astype(float)),
                     "first_week": str(min(i2).date()) if len(i2) else None,
                     "last_week": str(max(i2).date()) if len(i2) else None}
                 for k, (i2, i3) in periods.items()}
    contrasts = {"W_release_minus_pre_release": {
                     "h2": diff_desc(by_period["W_release"]["h2_weekly_spearman"],
                                     by_period["pre_release_complement"]["h2_weekly_spearman"]),
                     "h3": diff_desc(by_period["W_release"]["h3_delta_ic"],
                                     by_period["pre_release_complement"]["h3_delta_ic"])},
                 "d1_flagged_minus_rest": {
                     "h2": diff_desc(by_period["d1_flagged_months"]["h2_weekly_spearman"],
                                     by_period["not_d1_flagged"]["h2_weekly_spearman"]),
                     "h3": diff_desc(by_period["d1_flagged_months"]["h3_delta_ic"],
                                     by_period["not_d1_flagged"]["h3_delta_ic"])}}
    monthly = []
    for m in sorted(set(mk(h2w.index)) | set(mk(dic.index))):
        a_ = h2w[mk(h2w.index) == m]
        b_ = dic[mk(dic.index) == m]
        monthly.append({"month": m, "h2_mean_weekly_spearman": float(a_.mean()) if len(a_) else None,
                        "h2_weeks": int(len(a_)), "h2_units": int(h2n[mk(h2n.index) == m].sum()),
                        "h3_mean_delta_ic": float(b_.mean()) if len(b_) else None, "h3_weeks": int(len(b_)),
                        "ic_primary": float(icw.loc[b_.index, "primary"].astype(float).mean()) if len(b_) else None,
                        "ic_codeonly": float(icw.loc[b_.index, "codeonly"].astype(float).mean()) if len(b_)
                        else None,
                        "d1_flagged_month": m in flagged_months, "in_W_release": m >= D1.month_key(rel0)})
    events = [{"date": DOCUMENTED_CUTOFF, "label": "documented knowledge cutoff (gpt-4.1-mini)"},
              {"date": str(C.windows(root)["W_clean"][0].date()), "label": "W_clean start"},
              {"date": SNAPSHOT_RELEASE, "label": "pinned snapshot release"},
              {"date": str(rel0.date()), "label": "W_release start"}]
    # figure-ready tables
    FIGDIR.mkdir(parents=True, exist_ok=True)
    wrows = [{"period": "W_pre slice (development run)", **_wrow(w)} for w in ppooled] + \
            [{"period": "registered probe period", **_wrow(w)} for w in pooled]
    srows = [{"period": "W_pre slice (development run)", **_srow(w)} for w in psingle] + \
            [{"period": "registered probe period", **_srow(w)} for w in single]
    for df, nm in ((pd.DataFrame(wrows), "e3_d1_windows.csv"), (pd.DataFrame(srows), "e3_d1_single_months.csv"),
                   (pd.DataFrame(monthly), "e3_h2h3_monthly.csv"), (pd.DataFrame(events), "e3_events.csv")):
        with open(FIGDIR / nm, "w") as fh:
            fh.write(f"# {X.LABEL}; generated by experiments/exploratory/e3_lookahead_by_month.py\n")
            df.to_csv(fh, index=False)
    payload = {
        "question": "look-ahead by month: D1 recall AUC with permutation null bands; H2/H3 before vs after the "
                    "snapshot release; relation to the documented cutoff",
        "d1": {"registered_probe_period": {"months": months, "pooled_windows": pooled, "single_months": single,
                                           "windows_differing_from_0_5": flagged,
                                           "months_pooled_into_differing_windows": flagged_months,
                                           "rule_record": d1["rule"],
                                           "source": "results/analysis/d1.json (registered; AUC and p re-derived)"},
               "w_pre_slice": {"months": pmonths, "pooled_windows": ppooled, "single_months": psingle,
                               "source": "results/analysis/W_pre_slice/d1.json (development run; reproduced)"},
               "settings": {"n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED, "alpha": D1.ALPHA,
                            "null_band": "2.5-97.5% quantiles of the name-permuted AUC (stats.name_permutation_auc)"}},
        "h2_h3": {"registered_sensitivities": {"h2_W_release": h2["sensitivities"]["W_release"],
                                               "h3_W_release": h3["sensitivities"]["W_release"],
                                               "h3_W_release_dm_logloss": h3["sensitivities"]["W_release_dm_logloss"],
                                               "source": "results/analysis/h2.json, h3.json (registered; repeated)"},
                  "by_period": by_period, "contrasts_descriptive": contrasts, "monthly": monthly},
        "dates": events,
        "figure_data": {k: C.rel(FIGDIR / v, root) for k, v in (("d1_windows", "e3_d1_windows.csv"),
                                                                ("d1_single_months", "e3_d1_single_months.csv"),
                                                                ("h2h3_monthly", "e3_h2h3_monthly.csv"),
                                                                ("events", "e3_events.csv"))},
        "consistency": cons}
    inputs = {**{f"probe answers {g}": root / "results/judgments" / g / "answers.parquet"
                 for g in D1.REGISTERED_PROBES},
              "d1": root / "results/analysis/d1.json", "d1 W_pre": pre_path, "h2": root / "results/analysis/h2.json",
              "h3": root / "results/analysis/h3.json", "universe": root / C.UNIVERSE_FILE}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="null bands, single months, the W_pre extension and the H2/H3 period splits were specified after "
                    "the D1, H2 and H3 outcomes")
    for w in pooled:
        print(f"{w['center']} AUC {w['auc']:.3f} null95 [{w['null_band_95'][0]:.3f}, {w['null_band_95'][1]:.3f}] "
              f"p {w['p_perm']:.4f}{' *' if w['differs_from_0_5'] else ''}")
    bp = by_period
    print(f"H2 pre-release {bp['pre_release_complement']['h2_weekly_spearman']['mean']:.4f} / W_release "
          f"{bp['W_release']['h2_weekly_spearman']['mean']:.4f}; H3 dIC pre {bp['pre_release_complement']['h3_delta_ic']['mean']:.4f}"
          f" / rel {bp['W_release']['h3_delta_ic']['mean']:.4f}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


def _wrow(w: dict) -> dict:
    return {"center": w["center"], "auc": w["auc"], "null95_low": w["null_band_95"][0],
            "null95_high": w["null_band_95"][1], "p_perm": w["p_perm"], "differs_from_0_5": w["differs_from_0_5"],
            "ci_desc_low": (w.get("ci_week_block_descriptive") or [None, None])[0],
            "ci_desc_high": (w.get("ci_week_block_descriptive") or [None, None])[1], "n": w["n"],
            "weeks": w["weeks"], "edge": w["edge"]}


def _srow(w: dict) -> dict:
    return {"month": w["month"], "auc": w.get("auc"), "null95_low": (w.get("null_band_95") or [None, None])[0],
            "null95_high": (w.get("null_band_95") or [None, None])[1], "p_perm": w.get("p_perm"),
            "ci_desc_low": (w.get("ci_week_block_descriptive") or [None, None])[0],
            "ci_desc_high": (w.get("ci_week_block_descriptive") or [None, None])[1], "n": w["n"], "weeks": w["weeks"]}


if __name__ == "__main__":
    sys.exit(main())
