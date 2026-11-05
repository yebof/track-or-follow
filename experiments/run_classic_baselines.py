"""Baselines B1-B10 of design_v1 section 13.3 on the common clock, universe, engine and costs.
Writes results/baselines_classic.json.

Conventions (section 9.4, V-m11, V-M2, V-m14):
- weekly decisions (last session of each ISO week), fills at the next open, 15 bps per side (SPY and cash: 0);
- cash accrues DFF/252 (FRED, point in time); Sharpe and Sortino on excess returns; IR against EW-U100;
- decisions use the point-in-time U_t only: no filter on next-session tradability; unfillable targets stay in cash
  (counted as `n_unfilled`); held M&A exits leave at the last close, any other delisting would raise;
- prices: adjusted bars cut at each symbol's first gap > 10 sessions (the same cut as the raw bars).

Blocks:
- volatility-managed EW-U100 and SPY (design_v2_delta PE-5): exposure w_d = min(1, c_d / RV_d) of the underlying book,
  RV_d = mean squared daily return over the 22 sessions to d, c_d = mean squared daily return over all sessions to d
  (`baselines.classic.vol_managed_weight`; point in time, no leverage), the rest in cash at DFF; 15 bps per side on
  traded notional (SPY included, since the scaling trades);
- classical B1-B6 and Random-20 (B7; 1,000 seeded weekly redraws, percentile of each book's excess Sharpe) over
  START..END. This period includes W_clean; it re-runs the declared prior look `results/baselines_classic.json`
  (section 13.6) under the final convention, as section 9.4 requires.
- learned B8 (logistic) and B9 (LightGBM) on the 14 code features, and the B10 dictionary signal (weekly IC of
  lm_neg), only on decisions BEFORE W_clean: P-full over 2020-01-03..2024-06-28 and a P-clean dry run on the W_pre
  slice (cold start 2023-04-07). Their W_clean rows need the test lock opened by the pre-registration
  (`pipeline.lock`); pass --wclean after the unlock.

Run: .venv/bin/python experiments/run_classic_baselines.py [--random 1000] [--jobs 6] [--wclean]
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.backtest.engine import PriceBook, run_backtest, weekly_decision_dates  # noqa: E402
from jevtrader.baselines.classic import (BASELINES, ew_member_returns, random_k, vol_managed,  # noqa: E402
                                         vol_managed_weight)
from jevtrader.baselines.ml import LearnedBaseline, standardized_panel  # noqa: E402
from jevtrader.baselines.text import LMDictionary, lm_feature_panel  # noqa: E402
from jevtrader.data import fundamentals as F  # noqa: E402
from jevtrader.data.news import NewsStore  # noqa: E402
from jevtrader.data.prices import build_panels, cut_at, load_bars  # noqa: E402
from jevtrader.data.universe import ma_exits  # noqa: E402
from jevtrader.eval.metrics import information_coefficient, summarize, t_stat_newey_west  # noqa: E402
from jevtrader.features import technical as T  # noqa: E402
from jevtrader.features.macro import MacroStore  # noqa: E402
from jevtrader.learn.labels import LabelStore  # noqa: E402

START, END = "2020-01-01", "2026-09-18"
W_CLEAN = ("2024-07-05", "2026-09-11")
PFULL = ("2020-01-03", "2024-06-28")  # decisions before W_clean (W_pre + W_buffer)
PCLEAN_DRY = ("2023-04-07", "2024-05-31")  # W_pre slice
CODE = T.CODE_FEATURES + F.CODE_FEATURES
_G: dict = {}  # shared state for forked Random-20 workers


def _random_sharpe(seed: int) -> tuple[float, float]:
    """(net, gross) excess Sharpe of one Random-20 path. Weekly redraws turn over ~80% of the book, so the net
    figure mostly measures cost drag; the gross one is the selection-luck null."""
    rng = np.random.default_rng(seed)
    tg = {d: random_k(_G["members"][d], 20, rng) for d in _G["dates"]}
    out = []
    for bps in (15.0, 0.0):
        r = run_backtest(_G["pb"], tg, cost_bps=bps, cash_rate_daily=_G["rf"], ma_exits=_G["ma"])
        out.append(summarize(r.daily_returns, rf=r.rf)["sharpe"])
    return out[0], out[1]


def metrics(res, bench: pd.Series | None) -> dict:
    m = summarize(res.daily_returns, rf=res.rf, benchmark=bench)
    m["avg_turnover_per_fill"] = float(res.turnover.mean())
    m["n_unfilled"] = int(len(res.meta["unfilled"]))
    m["delisted"] = [f"{r.symbol}:{r.kind}:{pd.Timestamp(r.day).date()}" for r in res.meta["delisted"].itertuples()]
    return m


def weekly_ic(scores: pd.DataFrame, labels: LabelStore, col: str, sign: float = 1.0) -> dict:
    lab = labels.table.assign(rel=labels.table.r - labels.table.m)
    j = scores.merge(lab[["t", "symbol", "rel"]], on=["t", "symbol"]).dropna(subset=[col, "rel"])
    ic = information_coefficient(j.pivot(index="t", columns="symbol", values=col) * sign,
                                 j.pivot(index="t", columns="symbol", values="rel"))
    return {"n_weeks": int(ic.notna().sum()), "ic_mean": float(ic.mean()), "ic_sd": float(ic.std(ddof=1)),
            "ic_t_nw4": t_stat_newey_west(ic, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--wclean", action="store_true", help="also run B8/B9/B10 on W_clean (needs the unlock)")
    args = ap.parse_args()
    t0 = time.time()

    raw, adj = load_bars("raw"), load_bars("adjusted")
    panels = build_panels(raw, adj)
    pb = PriceBook(cut_at(adj, panels.life_end))
    rf = MacroStore(ROOT / "data/raw/macro").cash_rate_daily(pb.sessions)
    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    uni["date"] = pd.to_datetime(uni.date)
    recon = sorted(uni.date.unique())
    by_recon = {d: g.symbol.tolist() for d, g in uni.groupby("date")}
    dates = weekly_decision_dates(pb.sessions, START, END)
    members = {d: by_recon[max(r for r in recon if r <= d)] for d in dates}
    exits = ma_exits(sorted(set(uni.symbol)), raw, ROOT / "data/raw/sec/submissions")
    ma = {s for s, k in exits.items() if k == "mna"}
    sector = T.SectorMap(panels.adjusted)

    # ---------------------------------------------------------------- features at every decision (as of d)
    snaps, code_raw = {}, {}
    fund = F.FundamentalsPanel(ROOT / "data/raw/sec/facts", sorted(set(uni.symbol)), panels)
    for d in dates:
        snap = T.technical_snapshot(panels.adjusted, d, members[d], sector_etf=sector.asof(d, members[d]),
                                    raw=panels.raw)
        snaps[d] = snap
        code_raw[d] = T.code_features(snap).join(fund.features(d, members[d]), how="outer").reindex(members[d])
    print(f"features ready ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- B1-B6
    targets = {name: {d: fn(snaps[d]) for d in dates} for name, fn in BASELINES.items()}
    targets["spy"] = {d: {"SPY": 1.0} for d in dates}
    targets["cash_dff"] = {d: {} for d in dates}
    # PE-5 volatility-managed books (exposure from returns dated <= d only)
    spy_r = pb.close["SPY"].pct_change(fill_method=None).dropna()
    vm_w = {"vm_equal_weight": {d: vol_managed_weight(ew_member_returns(pb.close, members[d], d), d) for d in dates},
            "vm_spy": {d: vol_managed_weight(spy_r, d) for d in dates}}
    targets["vm_equal_weight"] = {d: vol_managed(targets["equal_weight"][d], vm_w["vm_equal_weight"][d]) for d in dates}
    targets["vm_spy"] = {d: vol_managed({"SPY": 1.0}, vm_w["vm_spy"][d]) for d in dates}
    res = {name: run_backtest(pb, tg, cost_bps=0.0 if name in ("spy", "cash_dff") else 15.0, cash_rate_daily=rf,
                              ma_exits=ma) for name, tg in targets.items()}
    bench = res["equal_weight"].daily_returns
    out = {name: metrics(r, bench) for name, r in res.items()}
    for name, tg in targets.items():  # gross of costs, for the gross Random-20 comparison
        g = run_backtest(pb, tg, cost_bps=0.0, cash_rate_daily=rf, ma_exits=ma)
        out[name]["sharpe_gross"] = summarize(g.daily_returns, rf=g.rf)["sharpe"]

    # ---------------------------------------------------------------- B7 Random-20
    _G.update(pb=pb, rf=rf, ma=ma, members=members, dates=dates)
    with mp.get_context("fork").Pool(args.jobs) as pool:
        both = np.array(pool.map(_random_sharpe, range(args.random)))
    random20 = {"n": int(len(both)), "seeds": f"0..{len(both) - 1}", "k": 20, "redraw": "weekly"}
    for j, (tag, key) in enumerate((("net_15bps", "sharpe"), ("gross", "sharpe_gross"))):
        rnd = both[:, j]
        random20[tag] = {"sharpe_mean": float(np.nanmean(rnd)), "sharpe_p05": float(np.nanpercentile(rnd, 5)),
                         "sharpe_p50": float(np.nanpercentile(rnd, 50)), "sharpe_p95": float(np.nanpercentile(rnd, 95)),
                         "percentile_of_book": {k: float((rnd < v[key]).mean()) for k, v in out.items()}}
    print(f"classical + random done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- B8, B9, B10 (pre-W_clean only)
    labels = LabelStore.build(pb, dates, members, ma_exits=ma)
    X = standardized_panel(code_raw, CODE)
    learned = {}
    windows = {"P-full": PFULL, "P-clean-dryrun-Wpre": PCLEAN_DRY}
    if args.wclean:
        from jevtrader.pipeline.lock import assert_unlocked_for

        assert_unlocked_for([pd.Timestamp(W_CLEAN[0])], ROOT)
        windows["P-clean"] = W_CLEAN
    for proto, (a, b) in windows.items():
        ds = [d for d in dates if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
        # hold the last book until the next weekly decision date (its label period), no further
        end = weekly_decision_dates(pb.sessions, ds[-1] + pd.Timedelta(days=1), ds[-1] + pd.Timedelta(days=14))[0]
        ew = run_backtest(pb, {d: {s: 1 / len(members[d]) for s in members[d]} for d in ds}, 15.0, rf, end=end,
                          ma_exits=ma)
        for kind in ("logit", "lgbm"):
            mdl = LearnedBaseline(kind, proto, pd.Timestamp(a), CODE)
            tg, pr = mdl.run(X, labels, ds, members)
            r = run_backtest(pb, tg, 15.0, rf, end=end, ma_exits=ma)
            learned[f"{'B8' if kind == 'logit' else 'B9'}_{kind}/{proto}"] = {
                "window": [a, b], "n_decisions": len(ds), "n_cold": int(sum(1 for x in mdl.log if x[1] == 0)),
                "portfolio": metrics(r, ew.daily_returns), "prediction": weekly_ic(pr[~pr.cold], labels, "p")}
        print(f"learned {proto} done ({time.time() - t0:.0f}s)", flush=True)

    news = NewsStore(pd.read_parquet(ROOT / "data/processed/news_table.parquet"))
    lm_dates = [d for d in dates if pd.Timestamp(PFULL[0]) <= d <= pd.Timestamp(PFULL[1])]
    lm = lm_feature_panel(news, lm_dates, members, LMDictionary())
    b10 = {"window": list(PFULL), "coverage": float(lm.lm_neg.notna().mean()),
           "ic_of_minus_lm_neg": weekly_ic(lm, labels, "lm_neg", sign=-1.0)}

    result = {
        "conventions": {"cost_bps": 15, "cash": "DFF/252 (FRED, PIT)", "sharpe": "excess over cash",
                        "ir_benchmark": "equal_weight (EW-U100)", "tradability_filter": "none (V-M2)",
                        "delistings": sorted(f"{s}:{k}" for s, k in exits.items())},
        "start": START, "end": END, "n_decisions": len(dates),
        "metrics": out, "random20": random20,
        "vol_managed_exposure": {k: {"mean": float(np.mean(list(v.values()))),
                                     "share_below_1": float(np.mean([x < 1 for x in v.values()])),
                                     "min": float(min(v.values())), "window_sessions": 22, "cap": 1.0}
                                 for k, v in vm_w.items()},
        "learned_pre_wclean": learned, "lm_dictionary_b10_pre_wclean": b10,
        "notes": ("B8/B9/B10 on W_clean are not run before the pre-registration unlock (pipeline.lock); "
                  "P-clean-dryrun-Wpre exercises the P-clean protocol on the W_pre slice."),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/baselines_classic.json").write_text(json.dumps(result, indent=1, default=float))
    for name, m in out.items():
        print(f"{name:14s} ann {m['ann_return']:+.1%} vol {m['ann_vol']:.1%} SR(excess) {m['sharpe']:.2f} "
              f"MDD {m['max_drawdown']:.1%} TO {m['avg_turnover_per_fill']:.2f} "
              f"IR {m.get('information_ratio', float('nan')):.2f}"
              f" unfilled {m['n_unfilled']} pct(R20 net) {random20['net_15bps']['percentile_of_book'][name]:.2f}"
              f" pct(R20 gross) {random20['gross']['percentile_of_book'][name]:.2f}")
    for k, v in learned.items():
        print(k, f"SR {v['portfolio']['sharpe']:.2f} IR {v['portfolio']['information_ratio']:.2f} "
                 f"IC {v['prediction']['ic_mean']:+.4f} t {v['prediction']['ic_t_nw4']:.2f}")
    print("B10", b10)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
