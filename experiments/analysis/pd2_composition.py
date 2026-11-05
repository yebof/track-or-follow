#!/usr/bin/env python3
"""PD2: the learned pool vs simple pools, the best single program and the oracle; leave-one-program-out with refit
(prereg §4.1, §4.2; design_v2_delta MA-1, MA-8, X1).

    .venv/bin/python experiments/analysis/pd2_composition.py                       # registered: MAIN on W_clean
    .venv/bin/python experiments/analysis/pd2_composition.py --gen-id pre_blinded_emu_wpre_s30 --window W_pre_slice

**Pools.** The learned pool is the primary allocator (its P and its logit L). V-EW-log (equal-weight log pool of the
per-program as-of heads), V-best (the program with the best matured log loss so far) and V-oracle (evaluation only:
per stock-week the head closest to the realized label) are recomputed from the primary run's logged heads
(heads.parquet) and the window's matured labels with fusion.pools.run_pools, exactly as run_layer_b --composition does
(the recomputed IC must equal composition.json's; checked).

**Estimates (descriptive; none enters a test family).** For each pool: the weekly rank-IC mean with a NW(4) 95% CI
and the pooled log loss (eval.metrics clipping); paired contrasts on the common stock-weeks: the learned pool minus
V-EW-log and minus V-best, as the weekly IC difference (NW CI) and the Diebold-Mariano-style weekly log-loss
difference (positive = the learned pool has the lower loss; NW CI); the oracle-gap share (L_best - L_pool) /
(L_best - L_oracle) of every probability pool with a week-block bootstrap interval (B = 2,000, seed 0).
Claim register: "composition adds" may be stated only if the learned pool's log-loss difference against V-EW-log has
a CI entirely above 0; otherwise "no detectable difference from equal-weight pooling".

**Leave-one-program-out with refit (X1).** The registered V-drop-{technical, fundamentals, macro, news, sentinel,
policy} runs: the primary minus each, as the weekly IC difference and the DM log-loss difference with NW CIs.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "pd2"
DEFAULT_GEN = "main_emu_wclean_u100"
POOLS = ("V-EW-log", "V-best", "V-oracle")
AGENTS = ("technical", "fundamentals", "macro", "news", "sentinel", "policy")
N_BOOT, SEED = 2000, 0


def weekly(frame: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Per week: IC of key vs r - m (>= 5 names) and the mean clipped log loss of P."""
    from jevtrader.eval.metrics import LOGLOSS_EPS

    j = frame.merge(lab, on=["t", "symbol"], how="inner").dropna(subset=["rel", "y"])
    p = j["P"].clip(LOGLOSS_EPS, 1 - LOGLOSS_EPS)
    j = j.assign(ll=-(j["y"] * np.log(p) + (1 - j["y"]) * np.log(1 - p)))
    rows = []
    for t, g in j.groupby("t", sort=True):
        with warnings.catch_warnings():  # a cold week's constant key has no rank correlation (NaN, as Layer B)
            warnings.simplefilter("ignore")
            ic = g["key"].corr(g["rel"], method="spearman") if g["key"].notna().sum() >= 5 else np.nan
        rows.append({"t": t, "ic": ic, "ll": g["ll"].mean(), "n": len(g)})
    return pd.DataFrame(rows)


def pooled_ll(frame: pd.DataFrame, lab: pd.DataFrame) -> float:
    from jevtrader.eval.metrics import LOGLOSS_EPS, log_loss

    j = frame.merge(lab, on=["t", "symbol"]).dropna(subset=["y", "P"])
    return log_loss(j["P"].to_numpy(), j["y"].to_numpy(), LOGLOSS_EPS) if len(j) else float("nan")


def paired_weekly(a: pd.DataFrame, b: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """Contrasts of a vs b on their common stock-weeks: weekly IC difference and weekly log-loss difference."""
    keys = a[["t", "symbol"]].merge(b[["t", "symbol"]], on=["t", "symbol"])
    wa = weekly(a.merge(keys, on=["t", "symbol"]), lab).set_index("t")
    wb = weekly(b.merge(keys, on=["t", "symbol"]), lab).set_index("t")
    d = wa.join(wb, lsuffix="_a", rsuffix="_b", how="inner")
    ic = S.nw_test((d["ic_a"] - d["ic_b"]).dropna().to_numpy(), S.NW_LAGS, "two-sided")
    ll = S.nw_test((d["ll_b"] - d["ll_a"]).to_numpy(), S.NW_LAGS, "two-sided")
    return {"n_stock_weeks": len(keys), "n_weeks": len(d), "delta_ic": ic,
            "delta_log_loss_b_minus_a": ll, "log_loss_a": float(d["ll_a"].mean()), "log_loss_b": float(d["ll_b"].mean())}


def gap_share(frames: dict[str, pd.DataFrame], lab: pd.DataFrame, pool: str) -> dict:
    from jevtrader.eval.metrics import LOGLOSS_EPS
    from jevtrader.fusion.pools import oracle_gap_share

    keys = frames[pool][["t", "symbol"]]
    for v in ("V-best", "V-oracle"):
        keys = keys.merge(frames[v][["t", "symbol"]], on=["t", "symbol"])
    keys = keys.merge(lab.dropna(subset=["y"])[["t", "symbol"]], on=["t", "symbol"])
    sums = {}
    for v in (pool, "V-best", "V-oracle"):
        j = frames[v].merge(keys, on=["t", "symbol"]).merge(lab[["t", "symbol", "y"]], on=["t", "symbol"])
        p = j["P"].clip(LOGLOSS_EPS, 1 - LOGLOSS_EPS)
        j = j.assign(ll=-(j["y"] * np.log(p) + (1 - j["y"]) * np.log(1 - p)))
        sums[v] = j.groupby("t")["ll"].agg(["sum", "size"]).sort_index()
    n = sums[pool]["size"].to_numpy(dtype=float)  # the same keys in every pool, so the same count per week
    s = {v: sums[v]["sum"].to_numpy(dtype=float) for v in sums}

    def share(idx) -> float:
        tot = n[idx].sum()
        return oracle_gap_share(s[pool][idx].sum() / tot, s["V-best"][idx].sum() / tot,
                                s["V-oracle"][idx].sum() / tot)

    G = len(n)
    est = share(np.arange(G))
    rng = np.random.default_rng(SEED)
    boot = np.array([share(rng.integers(0, G, G)) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"share": est, "ci": [float(lo), float(hi)], "n_stock_weeks": len(keys), "n_weeks": G,
            "n_boot": N_BOOT, "seed": SEED, "bootstrap": "weeks resampled with replacement (percentile)"}


def analyse(prim: C.EvalRun, lab_store, pool_cfg, drops: dict[str, C.EvalRun | None], logged: dict | None) -> dict:
    from jevtrader.fusion.pools import evaluate_scores, run_pools

    heads = pd.read_parquet(prim.file("heads.parquet"))
    heads["t"] = pd.to_datetime(heads["t"])
    run = run_pools(heads, lab_store, pool_cfg, ledger=None, variants=POOLS)
    lab = lab_store.table[["t", "symbol", "r", "m", "y"]].assign(rel=lambda x: x["r"] - x["m"])
    pr = prim.learned.rename(columns={"L": "key"})[["t", "symbol", "key", "P"]]
    frames = {"primary": pr, **{v: run.scores[v][["t", "symbol", "key", "P"]] for v in POOLS}}
    out = {"pools": {}, "paired": {}, "oracle_gap_share": {}, "leave_one_out": {}, "consistency": {}}
    for v, f in frames.items():
        w = weekly(f, lab)
        ev = evaluate_scores(f, lab_store)
        out["pools"][v] = {"ic": S.nw_test(w["ic"].dropna().to_numpy(), S.NW_LAGS, "two-sided"),
                           "log_loss": ev.get("scores", {}).get("log_loss"), "n": ev.get("n"),
                           "resolution": ev.get("scores", {}).get("resolution")}
        if logged and v in logged and "ic_mean" in logged[v] and "ic_mean" in ev:
            out["consistency"][f"{v}_ic_equals_logged"] = abs(ev["ic_mean"] - logged[v]["ic_mean"])
    for v in ("V-EW-log", "V-best"):
        out["paired"][f"primary_vs_{v}"] = paired_weekly(frames["primary"], frames[v], lab)
    for v in ("primary", "V-EW-log"):
        out["oracle_gap_share"][v] = gap_share(frames, lab, v)
    ll = out["paired"]["primary_vs_V-EW-log"]["delta_log_loss_b_minus_a"]
    out["claim_composition_adds"] = bool(ll["ci"][0] is not None and np.isfinite(ll["ci"][0]) and ll["ci"][0] > 0)
    for a, run_d in drops.items():
        if run_d is None:
            out["leave_one_out"][a] = {"status": "not run"}
            continue
        d = run_d.learned.rename(columns={"L": "key"})[["t", "symbol", "key", "P"]]
        out["leave_one_out"][a] = {"status": "ok", "dir": C.rel(run_d.path), **paired_weekly(frames["primary"], d, lab)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--primary", default=None)
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "PD2")
    pdir = Path(a.primary) if a.primary else C.find_eval_dir(root, a.window, "primary", a.gen_id)
    prim = C.read_eval_run(pdir, guard, "PD2 primary")
    C.require_d1_start(prim.dates[0], root, "PD2", a.window)
    drops = {}
    for ag in AGENTS:
        found = C.eval_dirs(root, a.window, name=f"V-drop-{ag}", gen_id=a.gen_id)
        drops[ag] = C.read_eval_run(found[0], guard, f"PD2 V-drop-{ag}") if len(found) == 1 else None
    from jevtrader.pipeline.variants import pool_config

    px = C.load_prices(guard, start, end)
    labels = C.load_labels(px, prim.dates, {t: px.universe[t] for t in prim.dates})
    labels = labels.restrict(prim.dates[0], prim.dates[-1])
    hl = prim.config.get("allocator", {}).get("half_life_weeks", 52.0)
    comp = C.read_json(prim.file("composition.json")) if prim.file("composition.json").exists() else None
    res = analyse(prim, labels, pool_config(half_life_weeks=hl), drops, comp.get("summary") if comp else None)
    payload = {"run": {"dir": C.rel(pdir, root), "cfg_sha": prim.metrics.get("cfg_sha")}, **res,
               "settings": {"pools": POOLS, "nw_lags": S.NW_LAGS, "n_boot": N_BOOT, "seed": SEED,
                            "price_cap": px.cap}}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard,
                   inputs={"heads": prim.file("heads.parquet"), "predictions": prim.file("predictions.parquet"),
                           "composition": prim.file("composition.json")}, argv=argv)
    g = res["oracle_gap_share"]["primary"]
    print(f"PD2 {a.window}: oracle-gap share {g['share']:.3f} CI {g['ci']}; composition adds: "
          f"{res['claim_composition_adds']} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
