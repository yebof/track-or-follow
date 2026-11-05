#!/usr/bin/env python3
"""H3 (code C3): the S1-incremental cross-sectional signal, primary vs V-codeonly (prereg §3.3).

    .venv/bin/python experiments/analysis/h3_signal.py                               # registered: MAIN on W_clean
    .venv/bin/python experiments/analysis/h3_signal.py --gen-id pre_blinded_emu_wpre_s30 --window W_pre_slice

**Inputs.** Two Layer-B run directories of the same generation and window (results/eval/<cfg16>/<window>__<inputs12>/,
written by experiments/run_layer_b.py): the primary cell and the registered variant V-codeonly (same architecture,
learner, cap and books; the S1 features replaced by the code features plus a raw news count). Optionally the coverage
placebo run (B11/VC-7). Found by config name and generation, or passed explicitly.

**Contrast.** IC_t = Spearman over U_t of L_{i,t} with r_{i,t} - m_t (T1's relative open-to-open return), from the first
learned decision (eval.metrics.information_coefficient; weeks with >= 5 names, as Layer B scores itself).
dIC_t = IC_t(primary) - IC_t(V-codeonly) over the common learned weeks; one-sided (greater) Newey-West(4) t with a
t(n-1) reference (eval.metrics.paired_ic_diff) and a 95% NW interval. Reported alongside: the Diebold-Mariano-style
paired log loss of P (eval.metrics.dm_logloss; positive = primary lower loss); the split IC(L) - IC(L^T) (the untrusted
text) and IC(L^T) - IC(code-only) (the trusted typed judgments), both descriptive; the coverage placebo; the
sensitivities W_release, the first 13 weeks dropped (weeks from common.drop_start, a date from the configs:
2024-10-04 on W_clean, as H2; the first 13 DECISIONS of the window, not of the learned weeks) and the prior-look
stock-weeks excluded; the correlation of the two weekly IC series (the power input rho).

Consistency checks written to the output: the recomputed IC series equals the one Layer B logged in metrics.json (the
labels are rebuilt here from the bars, so this checks the whole label path); with the D1 rule applied, the primary
run must start at the rule's start (a moved start needs a new cold-started replay).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "h3"
DEFAULT_GEN = "main_emu_wclean_u100"
DROP_FIRST = 13
ALTERNATIVE = "greater"


def rel_returns(labels_table: pd.DataFrame) -> pd.DataFrame:
    lab = labels_table[["t", "symbol", "r", "m", "y"]].copy()
    lab["t"] = pd.to_datetime(lab["t"])
    lab["rel"] = lab["r"] - lab["m"]
    return lab


def ic_series(pred: pd.DataFrame, lab: pd.DataFrame, col: str = "L", exclude: set | None = None) -> pd.Series:
    """Weekly rank-IC of `col` with r - m over the learned rows (eval.metrics.information_coefficient)."""
    from jevtrader.eval.metrics import information_coefficient

    p = pred[~pred["cold"].astype(bool)][["t", "symbol", col]]
    j = p.merge(lab, on=["t", "symbol"], how="inner").dropna(subset=["rel", "y"])
    if exclude:
        j = j[[(t, s) not in exclude for t, s in zip(j["t"], j["symbol"])]]
    return information_coefficient(j.pivot(index="t", columns="symbol", values=col),
                                   j.pivot(index="t", columns="symbol", values="rel")).sort_index()


def paired(a: pd.Series, b: pd.Series, weeks=None, alternative: str = ALTERNATIVE) -> dict:
    """mean(a_t - b_t) over common weeks with the NW(4) test (eval.metrics.paired_ic_diff) and CI."""
    from jevtrader.eval.metrics import paired_ic_diff

    d = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if weeks is not None:
        d = d[d.index.isin(set(pd.to_datetime(list(weeks))))]
    base = paired_ic_diff(d["a"], d["b"], S.NW_LAGS)
    t = S.nw_test((d["a"] - d["b"]).to_numpy(), S.NW_LAGS, alternative)
    ok = len(d) > 2 and d["a"].std() > 0 and d["b"].std() > 0
    corr = float(np.corrcoef(d["a"], d["b"])[0, 1]) if ok else float("nan")
    return {"mean": base["mean"], "t": base["t_nw"], "p": t["p"] if alternative != "greater" else base["p_one_sided"],
            "alternative": alternative, "ci": t["ci"], "se": t["se"], "n_weeks": base["n_weeks"],
            "lags": S.NW_LAGS, "corr_weekly": corr, "mean_a": float(d["a"].mean()) if len(d) else None,
            "mean_b": float(d["b"].mean()) if len(d) else None}


def dm(pred_a: pd.DataFrame, pred_b: pd.DataFrame, lab: pd.DataFrame, weeks=None) -> dict:
    from jevtrader.eval.metrics import dm_logloss

    a = pred_a[~pred_a["cold"].astype(bool)][["t", "symbol", "P"]]
    b = pred_b[~pred_b["cold"].astype(bool)][["t", "symbol", "P"]]
    j = a.merge(b, on=["t", "symbol"], suffixes=("_a", "_b")).merge(lab, on=["t", "symbol"]).dropna(subset=["y"])
    if weeks is not None:
        j = j[j["t"].isin(set(pd.to_datetime(list(weeks))))]
    if not len(j):
        return {"n": 0}
    out = dm_logloss(j["P_a"], j["P_b"], j["y"], j["t"], S.NW_LAGS)
    eps = 1e-4
    la = -(j["y"] * np.log(j["P_a"].clip(eps, 1 - eps)) + (1 - j["y"]) * np.log((1 - j["P_a"]).clip(eps, 1 - eps)))
    lb = -(j["y"] * np.log(j["P_b"].clip(eps, 1 - eps)) + (1 - j["y"]) * np.log((1 - j["P_b"]).clip(eps, 1 - eps)))
    w = pd.DataFrame({"t": j["t"], "d": lb - la}).groupby("t")["d"].mean().sort_index()
    out["ci"] = S.nw_test(w.to_numpy(), S.NW_LAGS, "greater")["ci"]
    out["n"] = len(j)
    return out


def analyse(prim: C.EvalRun, code: C.EvalRun, lab: pd.DataFrame, *, release: pd.Timestamp, prior: set,
            placebo: C.EvalRun | None = None, drop_from: pd.Timestamp | None = None) -> dict:
    """`drop_from`: the drop-13 sensitivity's first week (common.drop_start, a date from the configs: 2024-10-04 on
    W_clean, unchanged by the D1 rule; None = the 14th decision of the primary run, synthetic tests only)."""
    ic_p, ic_c = ic_series(prim.predictions, lab), ic_series(code.predictions, lab)
    ic_pt = ic_series(prim.predictions, lab, "L_T")
    weeks = sorted(set(ic_p.dropna().index) & set(ic_c.dropna().index))
    if drop_from is None:  # synthetic runs: the run's own 14th decision (none: every week is dropped)
        ds = sorted(prim.dates)
        drop_from = ds[DROP_FIRST] if len(ds) > DROP_FIRST else ds[-1] + pd.Timedelta(days=1)
    drop13 = pd.Timestamp(drop_from)
    conf = paired(ic_p, ic_c)
    out = {
        "confirmatory": {"hypothesis": "H3", "code": "C3", "contrast": "IC(primary) - IC(V-codeonly)", **conf},
        "ic": {"primary": S.nw_test(ic_p.dropna().to_numpy(), S.NW_LAGS, "two-sided"),
               "codeonly": S.nw_test(ic_c.dropna().to_numpy(), S.NW_LAGS, "two-sided")},
        "dm_logloss": dm(prim.predictions, code.predictions, lab),
        "split": {"text_part_IC_L_minus_IC_LT": paired(ic_p, ic_pt, alternative="two-sided"),
                  "trusted_typed_part_IC_LT_minus_IC_codeonly": paired(ic_pt, ic_c, alternative="two-sided")},
        "sensitivities": {
            "W_release": paired(ic_p, ic_c, [w for w in weeks if w >= release]),
            f"drop_first_{DROP_FIRST}": {**paired(ic_p, ic_c, [w for w in weeks if w >= drop13]), "from": drop13,
                                         "kind": "sub-series of the primary and V-codeonly runs (weeks >= from)"},
            "prior_look_excluded": paired(ic_series(prim.predictions, lab, exclude=prior),
                                          ic_series(code.predictions, lab, exclude=prior)),
            "W_release_dm_logloss": dm(prim.predictions, code.predictions, lab, [w for w in weeks if w >= release]),
        },
        "coverage_placebo": {"status": "not run"} if placebo is None else
        {"status": "ok", "delta_ic_over_placebo": paired(ic_p, ic_series(placebo.predictions, lab))},
        "consistency": {},
        "ic_by_week": {str(t.date()): {"primary": ic_p.get(t), "codeonly": ic_c.get(t), "primary_LT": ic_pt.get(t)}
                       for t in sorted(set(ic_p.index) | set(ic_c.index))},
    }
    for name, run, ic in (("primary", prim, ic_p), ("codeonly", code, ic_c)):
        logged = pd.Series({pd.Timestamp(k): v for k, v in run.metrics.get("prediction", {}).get("ic_by_week",
                                                                                                  {}).items()})
        both = pd.concat([ic.rename("a"), logged.rename("b")], axis=1, join="inner").dropna()
        out["consistency"][f"{name}_ic_equals_layer_b"] = {
            "weeks_compared": len(both), "weeks_logged": int(logged.notna().sum()),
            "max_abs_diff": float((both["a"] - both["b"]).abs().max()) if len(both) else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--primary", default=None, help="Layer-B run dir of the primary cell (default: found by name)")
    ap.add_argument("--codeonly", default=None, help="Layer-B run dir of V-codeonly (default: found by name)")
    ap.add_argument("--coverage-placebo", default=None, help="Layer-B run dir of the B11 coverage placebo (optional)")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    guard.check_window(a.window, "H3")
    pdir = Path(a.primary) if a.primary else C.find_eval_dir(root, a.window, "primary", a.gen_id)
    cdir = Path(a.codeonly) if a.codeonly else C.find_eval_dir(root, a.window, "V-codeonly", a.gen_id)
    prim = C.read_eval_run(pdir, guard, "H3 primary")
    code = C.read_eval_run(cdir, guard, "H3 V-codeonly")
    plac = C.read_eval_run(Path(a.coverage_placebo), guard, "H3 coverage placebo") if a.coverage_placebo else None
    if prim.dates != code.dates:
        raise SystemExit("H3: the primary and V-codeonly runs have different decision dates")
    d1 = C.require_d1_start(prim.dates[0], root, "H3 (replay the primary cell and V-codeonly cold-started at the "
                            "D1 start)", a.window)
    start, end = prim.dates[0], prim.dates[-1]
    px = C.load_prices(guard, start, end)
    uni = {t: tuple(prim.predictions.loc[prim.predictions["t"] == t, "symbol"]) for t in prim.dates}
    mism = {str(t.date()): sorted(set(px.universe.get(t, ())) ^ set(u)) for t, u in uni.items()
            if set(px.universe.get(t, ())) != set(u)}
    labels = C.load_labels(px, prim.dates, {t: px.universe.get(t, u) for t, u in uni.items()})
    lab = rel_returns(labels.table)
    res = analyse(prim, code, lab, release=C.release_start(root), prior=C.prior_look_keys(root), placebo=plac,
                  drop_from=C.drop_start(a.window, root))
    res["consistency"]["universe_mismatch_weeks"] = mism
    res["consistency"]["d1_rule"] = {k: v for k, v in d1.items()}
    payload = {"runs": {"primary": {"dir": C.rel(pdir, root), "cfg_sha": prim.metrics.get("cfg_sha"),
                                    "inputs_sha": prim.metrics.get("inputs_sha")},
                        "codeonly": {"dir": C.rel(cdir, root), "cfg_sha": code.metrics.get("cfg_sha"),
                                     "inputs_sha": code.metrics.get("inputs_sha")}},
               **res, "settings": {"nw_lags": S.NW_LAGS, "alternative": ALTERNATIVE, "drop_first": DROP_FIRST,
                                   "price_cap": px.cap}}
    inputs = {f"{k} {f}": d / f for k, d in (("primary", pdir), ("codeonly", cdir))
              for f in ("predictions.parquet", "metrics.json", "config.json")}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
    c = res["confirmatory"]
    print(f"H3 {a.window}: mean dIC {c['mean']:.4f} CI {c['ci']} t {c['t']:.2f} p {c['p']:.4g} "
          f"({c['n_weeks']} weeks) -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
