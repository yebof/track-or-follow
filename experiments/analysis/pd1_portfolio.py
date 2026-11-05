#!/usr/bin/env python3
"""PD1: the compact portfolio table with Sharpe CIs, factor attribution, DSR and the Random-20 percentile
(prereg §4.1, §4.4; design_v2_delta PE-2, PE-3, PE-5, PE-7, PE-8, PE-16).

    .venv/bin/python experiments/analysis/pd1_portfolio.py                          # registered: MAIN on W_clean
    .venv/bin/python experiments/analysis/pd1_portfolio.py --gen-id pre_blinded_emu_wpre_s30 --window W_pre_slice

Estimation only (portfolio IR MDE about 1.9): nothing here is a test.

**Rows (PE-16, <= 8).** D-sized (system book) and D-sel-EW (signal book) of the primary cell; EW-U100; SPY;
volatility-managed EW-U100 (PE-5: exposure min(1, c_d / RV_d) of the EW book, baselines.classic); momentum 12-1 (the
top 20 of U_t by the trusted code feature mom_12_1 = the 12-1 month return); LightGBM on the 14 code features under
P-clean (B9, baselines.ml.LearnedBaseline, cold start at the window start, matured labels only); the code-only
system (V-codeonly, its D-sized book). All on the primary cell's decision dates, filled at the next open, 15 bps per
side, cash at DFF, held to the primary cell's holding end (the engine and conventions of layer_b.evaluate). The
system books are re-run from their logged weights, and the returns must equal the run's daily_returns.parquet
(checked; the difference is written).

**Columns.** Annualized excess return (mean daily excess over DFF x 252); volatility of the excess return; Sharpe
with the studentized circular block-bootstrap 95% interval (stats.sharpe_ci; block 5, B = 4,999, seed 0); maximum
drawdown of the NAV; one-way turnover per rebalance (half the traded notional); break-even cost against EW-U100
(eval.metrics.break_even_cost, bps per side); own-factor selection alpha with its NW(5) 95% interval
(eval.attribution.style_attribution on the PE-3 point-in-time factors built from the generation's code features).

**Caption numbers.** DSR (bailey2014deflated) of each headline book over the REGISTERED trial set (prereg §4.4):
the primary's two books, the 29 configuration variants x 2 books and the 13 composition variants
(pipeline.variants.registered_names), M = 73. Rows of results/trials.jsonl outside that set are dropped (listed), and
a name registered more than once (a rerun) keeps its LATEST row; on the registered window every registered name
must be present except the declared not-run V-reader-HS/* (HS is cut), and raw M stays 73 (the registered count;
the two HS series count although they did not run, which only deflates more). The trial Sharpe variance comes from
the present rows. N-hat = rho-bar + (1 - rho-bar) M (Eq. 9, p.14), with rho-bar the mean pairwise correlation of the
same set's stored daily series (the primary and the configuration variants; composition variants store none). Both
the raw-M and the N-hat DSR are reported (make_numbers: dsr, dsrNhat); also over the headline-candidate set
{primary D-sized, primary D-sel-EW} (PE-1). Random-20 (B7): 1,000 seeded weekly redraws of 20 names of U_t (seeds 0..999, numpy default_rng), net of
15 bps; the percentile = the share of random excess Sharpes below the book's.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "pd1"
DEFAULT_GEN = "main_emu_wclean_u100"
COST_BPS = 15.0
K_BOOK = 20
N_RANDOM, RANDOM_K = 1000, 20
FULL_HISTORY = pd.Timedelta(days=366 * 12)  # every bar before the window (the archive starts in 2018)
SHARPE_BLOCK, SHARPE_BOOT, SEED = 5, 4999, 0
ROWS = ("sized", "sel_ew", "ew_u100", "spy", "vm_ew_u100", "momentum_12_1", "lightgbm", "codeonly")
LABELS = {"sized": "System book (D-sized)", "sel_ew": "Signal book (D-sel-EW)", "ew_u100": "EW-U100", "spy": "SPY",
          "vm_ew_u100": "Vol-managed EW-U100", "momentum_12_1": "Momentum 12-1", "lightgbm": "LightGBM (code features)",
          "codeonly": "Code-only system"}
_G: dict = {}


def targets_from_weights(run: C.EvalRun) -> dict[str, dict]:
    w = pd.read_parquet(run.file("weights.parquet"))
    w["t"] = pd.to_datetime(w["t"])
    out: dict[str, dict] = {}
    for (book, t), g in w.groupby(["book", "t"]):
        out.setdefault(book, {})[pd.Timestamp(t)] = dict(zip(g["symbol"], g["w"].astype(float)))
    return out


def backtest(px: C.Prices, tg: dict, rf, hold_end, ma, cost=COST_BPS):
    from jevtrader.backtest.engine import run_backtest

    return run_backtest(px.book, tg, cost_bps=cost, cash_rate_daily=rf, delisting_returns={}, end=hold_end,
                        ma_exits=ma)


def _inf_safe(x):
    """JSON has no infinity: a book that trades no more than EW-U100 has break-even +inf (ahead) or -inf (behind)."""
    return ("+inf" if x > 0 else "-inf") if isinstance(x, float) and np.isinf(x) else x


def row_metrics(res, rf: pd.Series, bench, factors: pd.DataFrame | None, n_boot: int = SHARPE_BOOT) -> dict:
    from jevtrader.eval import metrics as M
    from jevtrader.eval.attribution import style_attribution

    r = res.daily_returns
    ex = r - rf.reindex(r.index).fillna(0.0)
    nav = (1 + r.fillna(0.0)).cumprod()
    out = {"n_days": len(r), "ann_excess_return": float(ex.mean() * 252), "ann_vol": float(ex.std(ddof=1) *
                                                                                                  np.sqrt(252)),
           "sharpe": S.sharpe_ci(ex, SHARPE_BLOCK, n_boot, SEED), "max_drawdown": M.max_drawdown(nav),
           "turnover_one_way": float(0.5 * res.turnover.mean()) if len(res.turnover) else 0.0,
           "break_even_bps_vs_ew": _inf_safe(M.break_even_cost(r, res.turnover, bench.daily_returns, bench.turnover,
                                                                COST_BPS)) if bench is not None else None}
    if factors is not None:
        att = style_attribution(r, rf, factors)
        d = att.get("daily", {})
        out["alpha"] = {"annualized": d.get("alpha_annualized"), "ci": d.get("alpha_ci95_annualized"),
                        "p_two_sided": d.get("alpha_p_two_sided"), "r2": d.get("r2"), "n": d.get("n"),
                        "weekly_annualized": att.get("weekly", {}).get("alpha_annualized"),
                        "weekly_ci": att.get("weekly", {}).get("alpha_ci95_annualized"), "factors": att.get("factors")}
    return out


def _random_sharpe(seed: int) -> float:
    from jevtrader.backtest.engine import run_backtest
    from jevtrader.baselines.classic import random_k
    from jevtrader.eval.metrics import sharpe

    rng = np.random.default_rng(seed)
    tg = {d: random_k(list(_G["members"][d]), RANDOM_K, rng) for d in _G["dates"]}
    r = run_backtest(_G["pb"], tg, cost_bps=COST_BPS, cash_rate_daily=_G["rf"], end=_G["end"], ma_exits=_G["ma"])
    return sharpe(r.daily_returns, rf=r.rf)


def random20(px: C.Prices, dates, universe, rf, hold_end, ma, n: int, jobs: int) -> np.ndarray:
    _G.update(pb=px.book, members=universe, dates=list(dates), rf=rf, end=hold_end, ma=ma)
    if jobs <= 1 or n < 8:
        return np.array([_random_sharpe(s) for s in range(n)])
    with mp.get_context("fork").Pool(jobs) as pool:
        return np.array(pool.map(_random_sharpe, range(n)))


def baseline_targets(px: C.Prices, dates, universe, code_raw: pd.DataFrame, labels) -> dict[str, dict]:
    from jevtrader.baselines.classic import (
        equal_weight,
        ew_member_returns,
        top_k_scores,
        vol_managed,
        vol_managed_weight,
    )
    from jevtrader.baselines.ml import LearnedBaseline, standardized_panel
    from jevtrader.data.fundamentals import CODE_FEATURES as F_CODE
    from jevtrader.features.technical import CODE_FEATURES as T_CODE

    snaps = {d: pd.DataFrame(index=list(universe[d])) for d in dates}
    ew = {d: equal_weight(snaps[d]) for d in dates}
    vm = {d: vol_managed(ew[d], vol_managed_weight(ew_member_returns(px.book.close, list(universe[d]), d), d))
          for d in dates}
    cr = code_raw.copy()
    cr["t"] = pd.to_datetime(cr["t"])
    mom = cr[cr["feature"] == "mom_12_1"].pivot_table(index="t", columns="symbol", values="value", aggfunc="first")
    momentum = {d: top_k_scores(mom.loc[d].reindex(list(universe[d])) if d in mom.index
                                else pd.Series(dtype=float), K_BOOK) for d in dates}
    code = tuple(T_CODE) + tuple(F_CODE)
    wide = {d: cr[cr["t"] == d].pivot_table(index="symbol", columns="feature", values="value", aggfunc="first")
            .reindex(list(universe[d])) for d in dates}
    X = standardized_panel(wide, code)
    lgbm, _ = LearnedBaseline("lgbm", "P-clean", pd.Timestamp(dates[0]), code, k=K_BOOK, seed=SEED).run(
        X, labels, list(dates), {d: list(universe[d]) for d in dates})
    return {"ew_u100": ew, "spy": {d: {"SPY": 1.0} for d in dates}, "vm_ew_u100": vm, "momentum_12_1": momentum,
            "lightgbm": lgbm}


PRIMARY_TRIALS = ("primary/sel_ew", "primary/sized")
NOT_RUN_TRIALS = ("V-reader-HS/sel_ew", "V-reader-HS/sized")  # HS is cut (prereg §4.4, §10)


def registered_trial_names() -> list[str]:
    from jevtrader.pipeline.variants import registered_names

    return [*PRIMARY_TRIALS, *registered_names()]


M_REGISTERED = 73  # 2 + 29 x 2 + 13 (prereg §4.4); asserted against registered_trial_names() in the tests


def registry_trials(root: Path, window: str, gen_id: str, strict: bool = False) -> tuple[pd.DataFrame, dict]:
    """The window's registered trials (module doc): rows of the registered set only, the latest row per name.
    `strict` (the registered window): every registered name except NOT_RUN_TRIALS must be present."""
    from jevtrader.pipeline.registry import load_trials

    names = registered_trial_names()
    info = {"m_registered": len(names), "not_run_declared": list(NOT_RUN_TRIALS)}
    p = root / "results/trials.jsonl"
    if not p.exists():
        return pd.DataFrame(), {**info, "m_present": 0, "missing": names}
    t = load_trials(p)
    t = t[(t["window"] == window) & (t.get("gen_id", gen_id) == gen_id)]
    outside = sorted(set(t["name"]) - set(names))
    t = t[t["name"].isin(names)]
    order = "ts_utc" if "ts_utc" in t else None
    n_all = len(t)
    t = (t.sort_values(order, kind="stable") if order else t).drop_duplicates("name", keep="last")
    missing = [n for n in names if n not in set(t["name"])]
    info.update({"m_present": len(t), "missing": missing, "outside_set_dropped": outside,
                 "reruns_dropped": n_all - len(t)})
    if strict and set(missing) - set(NOT_RUN_TRIALS):
        raise SystemExit(f"PD1: registered trials missing from results/trials.jsonl: "
                         f"{sorted(set(missing) - set(NOT_RUN_TRIALS))}")
    return t.reset_index(drop=True), info


def trial_series(root: Path, window: str, trials: pd.DataFrame) -> pd.DataFrame:
    """Daily returns of the given trials whose series is stored: the run directory named by the trial's cfg_sha and
    inputs_sha (results/eval/<cfg16>/<window>__<inputs12>/daily_returns.parquet), column = the trial's book."""
    cols = {}
    for name, cfg, inp in trials[["name", "cfg_sha", "inputs_sha"]].itertuples(index=False):
        f = root / "results/eval" / str(cfg)[:16] / f"{C.window_tag(window)}__{str(inp)[:12]}" / "daily_returns.parquet"
        book = name.rsplit("/", 1)[1]
        if f.exists():
            r = pd.read_parquet(f)
            if book in r:
                cols[name] = r[book]
    return pd.DataFrame(cols)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--primary", default=None)
    ap.add_argument("--codeonly", default=None)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sharpe-boot", type=int, default=SHARPE_BOOT)
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "PD1")
    pdir = Path(a.primary) if a.primary else C.find_eval_dir(root, a.window, "primary", a.gen_id)
    cdir = Path(a.codeonly) if a.codeonly else C.find_eval_dir(root, a.window, "V-codeonly", a.gen_id)
    prim = C.read_eval_run(pdir, guard, "PD1 primary")
    code = C.read_eval_run(cdir, guard, "PD1 V-codeonly")
    C.require_d1_start(prim.dates[0], root, "PD1", a.window)
    gen = root / "results/judgments" / a.gen_id
    dates = prim.dates
    hold_end = pd.Timestamp(prim.metrics["hold_end"])
    # full history before the window: PE-5's c_d is the mean squared return over ALL sessions up to d, as
    # run_classic_baselines (bars from 2018); the books' own returns do not depend on the earlier bars
    px = C.load_prices(guard, start, end, pad_before=FULL_HISTORY)
    from jevtrader.eval.attribution import factor_inputs, own_factor_returns
    from jevtrader.pipeline import inputs as I

    universe = {d: px.universe[d] for d in dates}
    ma = I.ma_exit_set(px.root, sorted({s for v in universe.values() for s in v}))
    rf = C.cash_rate(px)
    rf = rf if rf is not None else pd.Series(0.0, index=px.book.sessions)
    labels = C.load_labels(px, dates, universe)
    code_raw = pd.read_parquet(gen / "code_features.parquet")
    firm = I.load_firm_info(gen)
    fac = own_factor_returns(px.book, factor_inputs(code_raw.assign(t=pd.to_datetime(code_raw["t"])), firm), universe,
                             dates, end=hold_end, cash_rate_daily=rf, ma_exits=ma)
    tg = {**{b: tgt for b, tgt in targets_from_weights(prim).items() if b in ("sized", "sel_ew")},
          **baseline_targets(px, dates, universe, code_raw, labels),
          "codeonly": targets_from_weights(code)["sized"]}
    res = {k: backtest(px, tg[k], rf, hold_end, ma) for k in ROWS}
    logged = pd.read_parquet(prim.file("daily_returns.parquet"))
    check = {}
    for b, col in (("sized", "sized"), ("sel_ew", "sel_ew"), ("ew_u100", "ew_u100")):
        d = pd.concat([res[b].daily_returns.rename("a"), logged[col].rename("b")], axis=1)
        check[b] = {"max_abs_diff": float((d["a"] - d["b"]).abs().max()), "n_days": len(d),
                    "missing": int(d.isna().any(axis=1).sum())}
    rows = {k: {"label": LABELS[k], **row_metrics(res[k], rf, res["ew_u100"], fac, a.sharpe_boot)} for k in ROWS}
    trials, tinfo = registry_trials(root, a.window, a.gen_id, strict=C.is_registered(a.window, root))
    M = tinfo["m_registered"] if C.is_registered(a.window, root) else len(trials)
    series = trial_series(root, a.window, trials)
    nh = S.implied_independent_trials(series, M)
    srs = trials["sr_per_period"].astype(float).tolist() if len(trials) else []
    dsr = {}
    for b in ("sized", "sel_ew"):
        ex = res[b].daily_returns - rf.reindex(res[b].daily_returns.index).fillna(0.0)
        cand = trials[trials["name"].isin(list(PRIMARY_TRIALS))]["sr_per_period"].astype(float).tolist()
        dsr[b] = {"raw_M": S.deflated_sharpe(ex, srs, M), "n_hat": S.deflated_sharpe(ex, srs, nh["n_hat"]),
                  "headline_candidates": S.deflated_sharpe(ex, cand)}
    rnd = random20(px, dates, universe, rf, hold_end, ma, a.n_random, a.jobs)
    from jevtrader.eval.metrics import sharpe

    pct = {b: float(np.mean(rnd < sharpe(res[b].daily_returns, rf=res[b].rf))) for b in ("sized", "sel_ew")}
    payload = {"rows": rows, "order": list(ROWS), "window_dates": [str(dates[0].date()), str(dates[-1].date())],
               "hold_end": hold_end, "n_decisions": len(dates),
               "dsr": {"n_trials": M, "trials": tinfo, "series_for_rho": sorted(series.columns),
                       "implied": nh, "by_book": dsr},
               "random20": {"n": len(rnd), "seeds": f"0..{len(rnd) - 1}", "k": RANDOM_K,
                            "sharpe_p05": float(np.nanpercentile(rnd, 5)) if len(rnd) else None,
                            "sharpe_p50": float(np.nanpercentile(rnd, 50)) if len(rnd) else None,
                            "sharpe_p95": float(np.nanpercentile(rnd, 95)) if len(rnd) else None,
                            "percentile_of_book": pct},
               "consistency": {"system_books_equal_logged_returns": check,
                               "logged_attribution": C.read_json(prim.file("attribution.json")).get("own_factors")
                               if prim.file("attribution.json").exists() else None},
               "settings": {"cost_bps": COST_BPS, "sharpe_block": SHARPE_BLOCK, "sharpe_boot": a.sharpe_boot,
                            "seed": SEED, "n_random": a.n_random, "price_cap": px.cap}}
    for k in ("logged_attribution",):
        la = payload["consistency"][k]
        if la:
            payload["consistency"][k] = {b: {"alpha_annualized": v.get("daily", {}).get("alpha_annualized")}
                                         for b, v in la.items()}
    inputs = {"primary weights": prim.file("weights.parquet"), "primary returns": prim.file("daily_returns.parquet"),
              "codeonly weights": code.file("weights.parquet"), "code features": gen / "code_features.parquet",
              "trials": root / "results/trials.jsonl"}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
    s = rows["sized"]["sharpe"]
    print(f"PD1 {a.window}: D-sized Sharpe {s['sharpe']:.2f} CI {s['ci']}; DSR {dsr['sized']['raw_M']['dsr']:.3f} "
          f"over {len(trials)} trials; Random-20 pct {pct['sized']:.2f} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
