#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the PD3 stores-off replays over the MEM generations (prereg §4.1 PD3, §9 MEM).

    .venv/bin/python experiments/exploratory/pd3_stores_off.py        # -> results/exploratory/pd3_stores_off.json

Why exploratory. PD3's estimand is registered (prereg §4.1: ΔIC, Δ log loss and ΔIR of the full system vs the
all-stores-off composite, restricted to S30), and its analysis entry point `experiments/analysis/pd3_stores.py` is
hashed. The step between them, a Layer-B replay over the MEM generations, had no entry point at the commit
(precommit_report §4 item 6: "anything added must be hashed before the commit; otherwise it is 'not run'"), so the
registered PD3 status stays "not run" (results/analysis/pd3.json, unchanged). This file builds that replay after the
unlock, so every number it writes is exploratory.

Replays (all $0; no model call; `pipeline.layer_b.evaluate(register=False, write=False)`, so no trial row and no run
directory). Base: MAIN's primary configuration with the logged `coverage` waiver (DEV-1; cfg 4ea23931...) on W_clean.
  full        MAIN's inputs unchanged (checked equal to the registered primary run's predictions)
  mepi_off    MEM arm 1 (`mem_emu_mepi-off_wclean_s30`: N3 re-asked with no N2 and no M-epi reads or writes, MAIN's
              N01 replayed): its N23 features replace MAIN's for the S30 stock-weeks it covers (all 115 weeks)
  mwork_off   MEM arm 2 (`mem_emu_mwork-off_wclean-last98_s30`: T re-asked with the blackboard's market regime not
              available): its trusted T features (learn.features.features_from_answers, the same mapping Layer B uses)
              replace MAIN's for the S30 stock-weeks it covers (the last 98 weeks, 2024-11-01..2026-09-11)
  all_off     both replacements (the all-stores-off composite of PD3)
Every other feature, the risk table (the sentinel is N01-based and N01 is replayed) and all prices stay MAIN's. The
learner is fit on all U100 names as in production, so a replacement on S30 also moves the learned coefficients.

Estimates (the definitions of the registered pd3_stores.py, imported read-only: weekly rank-IC of L with r - m over
the S30 names, >= 5 names; the Diebold-Mariano-style weekly log-loss difference of P; IR of each run's book in excess
of its EW-U100 book with a paired circular block-bootstrap interval, block 5, B = 4,999, seed 0). Each comparison is
full minus stores-off on the weeks the stores-off arm covers (mepi_off: 113 learned weeks; mwork_off and all_off: the
learned weeks >= 2024-11-01; IR on the daily returns after that date). ΔIC and Δ log loss carry Newey-West(4) 95%
intervals. Estimates and intervals only: no test, no p-value.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402
import pd3_stores as P3  # noqa: E402  (registered estimand helpers, read-only)
import stats as S  # noqa: E402

NAME = "pd3_stores_off"
GEN = "main_emu_wclean_u100"
MEM_EPI = "mem_emu_mepi-off_wclean_s30"
MEM_WORK = "mem_emu_mwork-off_wclean-last98_s30"
T_FEATURES = ("x_trend", "t_pullback", "t_capitulation", "t_breakout", "t_overextended", "tech_view")


def replace_rows(raw: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """raw with every (t, symbol, feature) row of `new` replaced by `new`'s row (new keys are appended)."""
    key = ["t", "symbol", "feature"]
    new = new[raw.columns.intersection(new.columns)].copy()
    new["t"] = pd.to_datetime(new["t"])
    idx = pd.MultiIndex.from_frame(raw[key])
    drop = idx.isin(pd.MultiIndex.from_frame(new[key]))
    out = pd.concat([raw[~drop], new], ignore_index=True)
    out.attrs = dict(raw.attrs)
    return out, int(drop.sum())


def mem_frames(root: Path, industry, clip: float) -> dict[str, pd.DataFrame]:
    from jevtrader.learn.features import features_from_answers
    from jevtrader.pipeline.layer_b import load_judgment_features

    epi = load_judgment_features(root / "results/judgments" / MEM_EPI, industry, clip)
    main_prog = pd.read_parquet(root / "results/judgments" / GEN / "features.parquet", columns=["feature", "program"])
    n23 = set(main_prog.loc[main_prog["program"] == "N23", "feature"])
    epi = epi[epi["feature"].isin(n23)].reset_index(drop=True)
    ans = pd.read_parquet(root / "results/judgments" / MEM_WORK / "answers.parquet")
    work = features_from_answers(ans, industry, list(T_FEATURES), clip=clip)
    return {"mepi_off": epi, "mwork_off": work}


def pred_of(res) -> pd.DataFrame:
    p = res.log.predictions.copy()
    p["t"] = pd.to_datetime(p["t"])
    return p


def compare(full_pred, off_pred, full_ret, off_ret, lab, subset, root, t0=None) -> dict:
    pf, po = P3.subset_rows(full_pred, subset, root), P3.subset_rows(off_pred, subset, root)
    if t0 is not None:
        pf, po = pf[pf["t"] >= t0], po[po["t"] >= t0]
    keys = pf[["t", "symbol"]].merge(po[["t", "symbol"]], on=["t", "symbol"])
    pf, po = pf.merge(keys, on=["t", "symbol"]), po.merge(keys, on=["t", "symbol"])
    ic_f, ic_o = P3.weekly_ic(pf, lab), P3.weekly_ic(po, lab)
    d = pd.concat([ic_f.rename("f"), ic_o.rename("o")], axis=1, join="inner").dropna()
    j = pf[~pf["cold"].astype(bool)][["t", "symbol", "P"]].merge(
        po[~po["cold"].astype(bool)][["t", "symbol", "P"]], on=["t", "symbol"], suffixes=("_f", "_o")).merge(
        lab[["t", "symbol", "y"]], on=["t", "symbol"]).dropna(subset=["y"])
    # eval.metrics.dm_logloss's weekly differences (clip 1e-4; d_t = loss(off) - loss(full), positive = full lower
    # loss, pd3_stores' sign), with a two-sided NW(4) interval instead of its one-sided p
    y = j["y"].to_numpy(dtype=float)

    def loss(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    wk = pd.DataFrame({"w": j["t"].to_numpy(), "lf": loss(j["P_f"]), "lo": loss(j["P_o"])}).groupby("w").mean()
    dm = {**S.nw_test((wk["lo"] - wk["lf"]).to_numpy(), S.NW_LAGS, "two-sided"),
          "loss_full": float(wk["lf"].mean()) if len(wk) else None,
          "loss_off": float(wk["lo"].mean()) if len(wk) else None,
          "sign": "loss(off) - loss(full); positive = the full system has the lower loss"}
    ra, rb = pd.DataFrame(full_ret), pd.DataFrame(off_ret)
    if t0 is not None:
        ra, rb = ra[ra.index > t0], rb[rb.index > t0]
    dic = S.nw_test((d["f"] - d["o"]).to_numpy(), S.NW_LAGS, "two-sided")
    strip = ("p", "t", "alternative")  # estimates and intervals only (module doc)
    return {"subset": subset, "first_week": None if t0 is None else str(pd.Timestamp(t0).date()),
            "n_stock_weeks": len(keys), "n_weeks_ic": len(d),
            "delta_ic": {**{k: v for k, v in dic.items() if k not in strip},
                         "ic_full": float(d["f"].mean()) if len(d) else None,
                         "ic_off": float(d["o"].mean()) if len(d) else None},
            "delta_log_loss": {k: v for k, v in dm.items() if k not in strip},
            "delta_ir_sized": P3.ir_diff(ra, rb, "sized"), "delta_ir_sel_ew": P3.ir_diff(ra, rb, "sel_ew")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", default="W_clean")
    ap.add_argument("--subset", default="S30")
    ap.add_argument("--allow-missing", default="coverage")
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "PD3 exploratory")
    from jevtrader.pipeline.layer_b import LEARNING_YAML, SYSTEM_YAML, DecisionConfig, evaluate, load_inputs

    allow = tuple(x for x in a.allow_missing.split(",") if x)
    cfg = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=GEN, name="primary", allow_missing=allow)
    base = load_inputs(cfg, start, end, root)
    frames = mem_frames(root, base.industry or {}, cfg.noul_clip)
    t0_work = pd.Timestamp(frames["mwork_off"]["t"].min())
    raws = {"full": (base.raw, 0)}
    r1, n1 = replace_rows(base.raw, frames["mepi_off"])
    r2, n2 = replace_rows(base.raw, frames["mwork_off"])
    r3, _ = replace_rows(r1, frames["mwork_off"])
    raws.update(mepi_off=(r1, n1), mwork_off=(r2, n2), all_off=(r3, n1 + n2))
    runs = {}
    for k, (raw, _n) in raws.items():
        res = evaluate(cfg, a.window, dataclasses.replace(base, raw=raw), root=root, register=False, write=False)
        runs[k] = res
        print(f"{k}: IC {res.metrics['prediction'].get('ic_mean'):.4f}", flush=True)
    # the in-memory full replay must equal the registered primary run
    prim = C.read_eval_run(C.find_eval_dir(root, a.window, "primary", GEN), guard, "PD3x primary")
    fp = pred_of(runs["full"])
    chk = fp.merge(prim.predictions[["t", "symbol", "L"]], on=["t", "symbol"], suffixes=("", "_reg"))
    check = {"n": len(chk), "max_abs_diff_L": float((chk["L"] - chk["L_reg"]).abs().max()),
             "registered_run": C.rel(prim.path, root)}
    px = C.load_prices(guard, start, end)
    labels = C.load_labels(px, prim.dates, {t: px.universe[t] for t in prim.dates})
    lab = labels.table[["t", "symbol", "r", "m", "y"]].assign(rel=lambda x: x["r"] - x["m"])
    out = {}
    for k, t0 in (("mepi_off", None), ("mwork_off", t0_work), ("all_off", t0_work)):
        out[k] = compare(fp, pred_of(runs[k]), runs["full"].returns, runs[k].returns, lab, a.subset, root, t0)
        out[k]["rows_replaced"] = raws[k][1]
        out[k]["ic_all_names"] = {"full": runs["full"].metrics["prediction"].get("ic_mean"),
                                  "off": runs[k].metrics["prediction"].get("ic_mean")}
    payload = {"status": "ok", "estimand": "prereg §4.1 PD3 (full minus stores off), S30, descriptive",
               "arms": {"mepi_off": MEM_EPI, "mwork_off": MEM_WORK, "all_off": f"{MEM_EPI} + {MEM_WORK}"},
               "full_replay_equals_registered_primary": check, "comparisons": out,
               "settings": {"nw_lags": S.NW_LAGS, "block": P3.BLOCK, "n_boot": P3.N_BOOT, "seed": P3.SEED,
                            "allow_missing": list(allow), "cfg_sha": cfg.sha, "mwork_first_week": str(t0_work.date())},
               "registered_status": "results/analysis/pd3.json stays 'not run' (no hashed stores-off builder)"}
    g = root / "results/judgments"
    inputs = {"MEM arm 1 features": g / MEM_EPI / "features.parquet", "MEM arm 2 answers": g / MEM_WORK /
              "answers.parquet", "MAIN features": g / GEN / "features.parquet",
              "primary predictions": prim.path / "predictions.parquet"}
    p = X.write(NAME, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv,
                why="the stores-off Layer-B replay over MEM had no hashed entry point at the commit; built after "
                    "the unlock")
    for k, v in out.items():
        di, dl = v["delta_ic"], v["delta_log_loss"]
        print(f"{k}: dIC {di['mean']:+.4f} {di['ci']}; dLL {dl.get('mean')} {dl.get('ci')}; "
              f"dIR sized {v['delta_ir_sized']['diff']:+.3f} {v['delta_ir_sized']['ci']}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
