#!/usr/bin/env python3
"""PD3 (secondary descriptive): the full system vs the all-stores-off composite (prereg §4.1; design_v2_delta MEM-1).

    .venv/bin/python experiments/analysis/pd3_stores.py --stores-off <Layer-B run dir of the MEM generation>
    .venv/bin/python experiments/analysis/pd3_stores.py --window W_pre_slice --full <dir> --stores-off <dir>

**Inputs.** Two Layer-B run directories on the same decision dates: the primary cell ("full", default: found by name
for the generation) and the replay over the MEM generation with every store off (M-epi off: N3 re-asked without the
event store; M-work off: T re-asked without the blackboard market field), passed with --stores-off. MEM runs as
reduced (prereg §9), so the comparison is restricted to the S30 names of U_t (--subset S30, the default; `none` keeps
every name) and, for M-work, to the weeks it covers (the common weeks of the two runs).

**Estimates (reported whatever their sign; no test).** dIC_t = IC_t(full) - IC_t(stores off), each over the subset's
names (rank-IC of L with r - m, >= 5 names), mean with a NW(4) 95% CI; the Diebold-Mariano-style weekly log-loss
difference of P on the same stock-weeks (positive = full lower loss; NW CI); dIR of the system books (D-sized):
IR = annualized mean / sd of the daily return in excess of the run's EW-U100 book, the difference full - off with a
circular block-bootstrap 95% interval of the difference (block 5 days, B = 4,999, seed 0).
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "pd3"
DEFAULT_GEN = "main_emu_wclean_u100"
BLOCK, N_BOOT, SEED = 5, 4999, 0


def subset_rows(pred: pd.DataFrame, subset: str | None, root: Path) -> pd.DataFrame:
    if not subset or subset == "none":
        return pred
    keep = []
    for t, g in pred.groupby("t"):
        names = set(C.subset(list(g["symbol"]), subset, root))
        keep.append(g[g["symbol"].isin(names)])
    return pd.concat(keep, ignore_index=True)


def weekly_ic(pred: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    j = pred[~pred["cold"].astype(bool)].merge(lab, on=["t", "symbol"]).dropna(subset=["rel"])
    out = {}
    for t, g in j.groupby("t", sort=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[t] = g["L"].corr(g["rel"], method="spearman") if len(g) >= 5 else np.nan
    return pd.Series(out, dtype=float)


def ir_diff(a: pd.DataFrame, b: pd.DataFrame, book: str = "sized") -> dict:
    """IR of `book` vs the run's EW-U100 for each run and the difference with a paired circular block bootstrap."""
    act = pd.concat([(a[book] - a["ew_u100"]).rename("a"), (b[book] - b["ew_u100"]).rename("b")], axis=1,
                    join="inner").dropna()
    x = act.to_numpy()
    T = len(x)

    def ir(v):
        sd = v.std(axis=-2, ddof=1)
        return v.mean(axis=-2) / np.where(sd > 0, sd, np.nan) * math.sqrt(252)

    base = ir(x)
    out = {"ir_full": float(base[0]), "ir_off": float(base[1]), "diff": float(base[0] - base[1]), "n_days": T}
    if T < 3 * BLOCK:
        return {**out, "ci": [None, None]}
    rng = np.random.default_rng(SEED)
    nb = int(math.ceil(T / BLOCK))
    starts = rng.integers(0, T, size=(N_BOOT, nb))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % T).reshape(N_BOOT, -1)[:, :T]
    bs = ir(x[idx])
    d = bs[:, 0] - bs[:, 1]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {**out, "ci": [float(lo), float(hi)], "block": BLOCK, "n_boot": N_BOOT, "seed": SEED}


def analyse(full: C.EvalRun, off: C.EvalRun, lab: pd.DataFrame, subset: str | None, root: Path) -> dict:
    pf, po = subset_rows(full.predictions, subset, root), subset_rows(off.predictions, subset, root)
    keys = pf[["t", "symbol"]].merge(po[["t", "symbol"]], on=["t", "symbol"])
    pf, po = pf.merge(keys, on=["t", "symbol"]), po.merge(keys, on=["t", "symbol"])
    ic_f, ic_o = weekly_ic(pf, lab), weekly_ic(po, lab)
    d = pd.concat([ic_f.rename("f"), ic_o.rename("o")], axis=1, join="inner").dropna()
    from jevtrader.eval.metrics import dm_logloss

    j = pf[~pf["cold"].astype(bool)][["t", "symbol", "P"]].merge(
        po[~po["cold"].astype(bool)][["t", "symbol", "P"]], on=["t", "symbol"], suffixes=("_f", "_o")).merge(
        lab[["t", "symbol", "y"]], on=["t", "symbol"]).dropna(subset=["y"])
    dm = dm_logloss(j["P_f"], j["P_o"], j["y"], j["t"], S.NW_LAGS) if len(j) else {"n_weeks": 0}
    ra = pd.read_parquet(full.file("daily_returns.parquet"))
    rb = pd.read_parquet(off.file("daily_returns.parquet"))
    return {"subset": subset or "none", "n_stock_weeks": len(keys),
            "delta_ic": {**S.nw_test((d["f"] - d["o"]).to_numpy(), S.NW_LAGS, "two-sided"),
                         "ic_full": float(d["f"].mean()) if len(d) else None,
                         "ic_off": float(d["o"].mean()) if len(d) else None},
            "delta_log_loss": dm,
            "delta_ir_sized": ir_diff(ra, rb, "sized"), "delta_ir_sel_ew": ir_diff(ra, rb, "sel_ew")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--full", default=None)
    ap.add_argument("--stores-off", default=None, help="Layer-B run dir over the MEM (all stores off) generation")
    ap.add_argument("--subset", default="S30")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "PD3")
    path = C.out_path(NAME, a.window, a.out_dir, root)
    if not a.stores_off:
        C.write_result(path, {"status": "not run", "reason": "no all-stores-off replay (MEM not run)"},
                       script=__file__, window=a.window, guard=guard, argv=argv)
        print(f"PD3 {a.window}: not run (no --stores-off) -> {C.rel(path, root)}")
        return 0
    fdir = Path(a.full) if a.full else C.find_eval_dir(root, a.window, "primary", a.gen_id)
    full = C.read_eval_run(fdir, guard, "PD3 full")
    off = C.read_eval_run(Path(a.stores_off), guard, "PD3 stores off")
    px = C.load_prices(guard, start, end)
    labels = C.load_labels(px, full.dates, {t: px.universe[t] for t in full.dates})
    lab = labels.table[["t", "symbol", "r", "m", "y"]].assign(rel=lambda x: x["r"] - x["m"])
    res = analyse(full, off, lab, a.subset, root)
    payload = {"status": "ok", "runs": {"full": C.rel(fdir, root), "stores_off": C.rel(Path(a.stores_off), root)},
               **res, "settings": {"nw_lags": S.NW_LAGS, "block": BLOCK, "n_boot": N_BOOT, "seed": SEED,
                                   "price_cap": px.cap}}
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard,
                   inputs={"full predictions": full.file("predictions.parquet"),
                           "off predictions": off.file("predictions.parquet")}, argv=argv)
    print(f"PD3 {a.window}: dIC {res['delta_ic']['mean']:.4f} CI {res['delta_ic']['ci']}; dIR(sized) "
          f"{res['delta_ir_sized']['diff']:.3f} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
