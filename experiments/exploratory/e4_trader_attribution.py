#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E4: the system as a trader. D-sized's weekly returns against EW-U100, split into
the trusted-only (L^T) selection, the sizing/risk overlay and the text's contribution; turnover and cost attribution.

    .venv/bin/python experiments/exploratory/e4_trader_attribution.py
        # -> results/exploratory/e4_trader_attribution.json + results/exploratory/figdata/e4_weekly_layers.csv

Why exploratory. PD1 (prereg §4.1; results/analysis/pd1.json, Table 4) reports each book's Sharpe, drawdown,
turnover, break-even cost and own-factor alpha; it does not attribute the system book's active return to its layers.
This split was specified after PD1's outcome was known. Estimates with two-sided Newey-West(4) 95% intervals on the
weekly series (stats.nw_test); no test, nothing decides.

Books (U100 x W_clean; every book from a REGISTERED Layer-B run directory's logged weights, re-run with PD1's engine
and conventions, pd1_portfolio.backtest: the primary cell's decision dates, filled at the next open, 15 bps per side
(and 0 bps for the gross series), cash at DFF, held to the primary cell's holding end; the primary books and EW-U100
must reproduce the run's daily_returns.parquet, checked):
- EW-U100 (baselines.classic.equal_weight over U_t);
- T-EW: V-J0's D-sel-EW = the trusted top-K by L^T, equal-weighted (J = 0: the text cannot move selection);
- T-sized: V-J0's D-sized = the same trusted-only selection with the full sizing/risk overlay (reference-anchored
  sizing, anchor-gated exclusion, caps, turnover limit, incumbents);
- S-EW: the primary D-sel-EW (J = 10: the text reorders the in-band names);
- S-sized: the primary D-sized (the system book).
Weekly returns compound each book's daily returns over each holding week (after d_t through the next decision).

Decomposition of S-sized - EW-U100 (weekly, bps), two orders (the split is order-dependent):
- order A (selection -> overlay -> text): trusted selection = T-EW - EW; overlay = T-sized - T-EW; text =
  S-sized - T-sized;
- order B (selection -> text -> overlay): trusted selection = T-EW - EW; text = S-EW - T-EW; overlay = S-sized - S-EW.
Each layer: mean net and gross (0 bps) weekly return with NW CI, annualized (x 52), its share of the active variance
(covariance with the total active return / its variance), and the cost of the layer = gross - net. Turnover: one-way
per rebalance (PD1's 0.5 x mean traded notional), annualized cost drag, invested share (sum of target weights).
The text's name-level contribution: per learned week, the names S-EW holds and T-EW does not (text entrants) and the
reverse (displaced trusted names), their count and the mean r - m_t (the T1 label, open-to-open, gross) of each set;
entrants minus displaced with a NW CI.
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
import pd1_portfolio as P1  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

NAME = "e4_trader_attribution"
GEN = "main_emu_wclean_u100"
FIGDIR = X.OUT_DIR / "figdata"
BOOKS = ("ew", "t_ew", "t_sized", "s_ew", "s_sized")
LABELS = {"ew": "EW-U100", "t_ew": "trusted top-K, EW (V-J0 D-sel-EW)", "t_sized": "trusted top-K, sized (V-J0 D-sized)",
          "s_ew": "system selection, EW (primary D-sel-EW)", "s_sized": "system book (primary D-sized)"}


def weekly(daily: pd.Series, dates, hold_end) -> pd.Series:
    r = daily.dropna()
    edges = [pd.Timestamp(d) for d in sorted(dates)]
    out = {}
    for i, d in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else pd.Timestamp(hold_end)
        m = (r.index > d) & (r.index <= hi)
        if m.any():
            out[d] = float((1 + r[m]).prod() - 1)
    return pd.Series(out, dtype=float)


def nw(x: pd.Series, scale: float = 1e4) -> dict:
    t = S.nw_test(pd.Series(x, dtype=float).dropna().to_numpy() * scale, S.NW_LAGS, "two-sided")
    return {"mean": t["mean"], "ci": t["ci"], "n_weeks": int(t["n"])}


def main(argv=None) -> int:
    from jevtrader.baselines.classic import equal_weight
    from jevtrader.pipeline import inputs as I

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window("W_clean", "E4")
    prim = C.read_eval_run(C.find_eval_dir(root, "W_clean", "primary", GEN), guard, "E4 primary")
    j0 = C.read_eval_run(C.find_eval_dir(root, "W_clean", "V-J0", GEN), guard, "E4 V-J0")
    dates = prim.dates
    hold_end = pd.Timestamp(prim.metrics["hold_end"])
    px = C.load_prices(guard, start, end, pad_before=P1.FULL_HISTORY)
    universe = {d: px.universe[d] for d in dates}
    ma = I.ma_exit_set(px.root, sorted({s for v in universe.values() for s in v}))
    rf = C.cash_rate(px)
    rf = rf if rf is not None else pd.Series(0.0, index=px.book.sessions)
    tp, tj = P1.targets_from_weights(prim), P1.targets_from_weights(j0)
    tg = {"ew": {d: equal_weight(pd.DataFrame(index=list(universe[d]))) for d in dates},
          "t_ew": tj["sel_ew"], "t_sized": tj["sized"], "s_ew": tp["sel_ew"], "s_sized": tp["sized"]}
    net = {k: P1.backtest(px, tg[k], rf, hold_end, ma) for k in BOOKS}
    gross = {k: P1.backtest(px, tg[k], rf, hold_end, ma, cost=0.0) for k in BOOKS}
    logged = pd.read_parquet(prim.file("daily_returns.parquet"))
    lj0 = pd.read_parquet(j0.file("daily_returns.parquet"))
    check = {f"{k}_vs_{col}": float((net[k].daily_returns - src[col].reindex(net[k].daily_returns.index)).abs().max())
             for k, src, col in (("s_sized", logged, "sized"), ("s_ew", logged, "sel_ew"), ("ew", logged, "ew_u100"),
                                 ("t_sized", lj0, "sized"), ("t_ew", lj0, "sel_ew"))}
    if max(check.values()) > 1e-12:
        raise SystemExit(f"E4: re-run books differ from the logged returns: {check}")
    wn = pd.DataFrame({k: weekly(net[k].daily_returns, dates, hold_end) for k in BOOKS})
    wg = pd.DataFrame({k: weekly(gross[k].daily_returns, dates, hold_end) for k in BOOKS})
    cold = prim.predictions.groupby("t")["cold"].first()
    learned = [t for t in wn.index if not bool(cold.get(t, True))]
    wn, wg = wn.loc[learned], wg.loc[learned]
    total_n, total_g = wn["s_sized"] - wn["ew"], wg["s_sized"] - wg["ew"]
    orders = {"A_selection_overlay_text": {"trusted_selection": ("t_ew", "ew"), "overlay": ("t_sized", "t_ew"),
                                           "text": ("s_sized", "t_sized")},
              "B_selection_text_overlay": {"trusted_selection": ("t_ew", "ew"), "text": ("s_ew", "t_ew"),
                                           "overlay": ("s_sized", "s_ew")}}
    decomp = {}
    for o, layers in orders.items():
        decomp[o] = {}
        for layer, (x, y) in layers.items():
            dn, dg = wn[x] - wn[y], wg[x] - wg[y]
            decomp[o][layer] = {"books": f"{x} - {y}", "net_bps_per_week": nw(dn), "gross_bps_per_week": nw(dg),
                                "cost_bps_per_week": nw(dg - dn),
                                "net_annualized": float(dn.mean() * 52), "gross_annualized": float(dg.mean() * 52),
                                "share_of_active_variance": float(np.cov(dn, total_n)[0, 1] / total_n.var(ddof=1)),
                                "hit_rate": float((dn > 0).mean())}
    # the overlay (order A) split into the uninvested share and the re-weighting of the invested part: if the cash
    # share 1 - sum(w) of T-sized had been invested in T-EW, the book would have earned (1 - inv)(T-EW - cash) more
    rfw = weekly(rf.reindex(px.book.sessions).fillna(0.0), dates, hold_end).reindex(wn.index)
    inv_t = pd.Series({t: sum(tg["t_sized"].get(t, {}).values()) for t in wn.index}, dtype=float)
    cash_part = -(1.0 - inv_t) * (wn["t_ew"] - rfw)
    ov = wn["t_sized"] - wn["t_ew"]
    decomp["A_selection_overlay_text"]["overlay_split"] = {
        "uninvested_share_part_bps_per_week": nw(cash_part), "reweighting_part_bps_per_week": nw(ov - cash_part),
        "uninvested_share_mean": float((1.0 - inv_t).mean()),
        "definition": "uninvested part = -(1 - sum of T-sized target weights) x (T-EW weekly return - cash); "
                      "re-weighting part = overlay - uninvested part (inverse-risk sizing, caps, exclusions, "
                      "incumbents); first-order split, drift within the week ignored"}
    total = {"net_bps_per_week": nw(total_n), "gross_bps_per_week": nw(total_g), "cost_bps_per_week":
             nw(total_g - total_n), "net_annualized": float(total_n.mean() * 52),
             "tracking_error_annualized": float(total_n.std(ddof=1) * np.sqrt(52)),
             "information_ratio_weekly_annualized": float(total_n.mean() / total_n.std(ddof=1) * np.sqrt(52))}
    books = {}
    for k in BOOKS:
        r0, r15 = gross[k].daily_returns, net[k].daily_returns
        inv = pd.Series({t: sum(v.values()) for t, v in tg[k].items() if t in set(learned)}, dtype=float)
        books[k] = {"label": LABELS[k], "turnover_one_way_per_rebalance": float(0.5 * net[k].turnover.mean()),
                    "cost_drag_annualized": float((r0 - r15).mean() * 252),
                    "mean_weekly_net_bps": nw(wn[k]), "mean_weekly_gross_bps": nw(wg[k]),
                    "ann_return_net": float(wn[k].mean() * 52), "ann_vol_weekly": float(wn[k].std(ddof=1) *
                                                                                         np.sqrt(52)),
                    "invested_share_mean": float(inv.mean()) if len(inv) else None,
                    "n_names_mean": float(np.mean([len(v) for t, v in tg[k].items() if t in set(learned)]))}
    # the text's name-level contribution on the signal book
    labels = C.load_labels(px, dates, universe)
    rel = (labels.table.set_index(["t", "symbol"])["r"] - labels.table.set_index(["t", "symbol"])["m"])
    rows = []
    for t in learned:
        s_ew, t_ew = set(tp["sel_ew"].get(t, {})), set(tj["sel_ew"].get(t, {}))
        ent, dis = sorted(s_ew - t_ew), sorted(t_ew - s_ew)
        re = [rel.get((t, s), np.nan) for s in ent]
        rd = [rel.get((t, s), np.nan) for s in dis]
        rows.append({"t": t, "n_entrants": len(ent), "n_displaced": len(dis),
                     "rel_entrants": float(np.nanmean(re)) if ent else np.nan,
                     "rel_displaced": float(np.nanmean(rd)) if dis else np.nan})
    nm = pd.DataFrame(rows).set_index("t")
    nm["diff"] = nm["rel_entrants"] - nm["rel_displaced"]
    names = {"n_weeks": int(len(nm)), "weeks_with_swaps": int((nm["n_entrants"] > 0).sum()),
             "entrants_per_week_mean": float(nm["n_entrants"].mean()),
             "rel_entrants_bps": nw(nm["rel_entrants"]), "rel_displaced_bps": nw(nm["rel_displaced"]),
             "entrants_minus_displaced_bps": nw(nm["diff"]),
             "share_of_signal_book_replaced": float(nm["n_entrants"].mean() / P1.K_BOOK),
             "implied_book_effect_bps": float((nm["diff"].fillna(0) * nm["n_entrants"] / P1.K_BOOK).mean() * 1e4),
             "definition": "entrants = in the primary D-sel-EW and not in V-J0's; displaced = the reverse; r - m_t = "
                           "the T1 label (open-to-open holding week, gross of costs)"}
    # figure-ready weekly layers (order A) and cumulative sums
    fig = pd.DataFrame({"t": [str(t.date()) for t in wn.index], "active_total_net": total_n.to_numpy() * 1e4,
                        "trusted_selection_net": (wn["t_ew"] - wn["ew"]).to_numpy() * 1e4,
                        "overlay_net_A": (wn["t_sized"] - wn["t_ew"]).to_numpy() * 1e4,
                        "text_net_A": (wn["s_sized"] - wn["t_sized"]).to_numpy() * 1e4,
                        "text_net_B": (wn["s_ew"] - wn["t_ew"]).to_numpy() * 1e4,
                        "overlay_net_B": (wn["s_sized"] - wn["s_ew"]).to_numpy() * 1e4})
    FIGDIR.mkdir(parents=True, exist_ok=True)
    with open(FIGDIR / "e4_weekly_layers.csv", "w") as fh:
        fh.write(f"# {X.LABEL}; generated by experiments/exploratory/e4_trader_attribution.py; weekly net returns in "
                 f"bps, D-sized minus EW-U100 split into layers (orders A and B)\n")
        fig.to_csv(fh, index=False)
    pd1 = C.read_json(root / "results/analysis/pd1.json")
    payload = {"question": "the system as a trader: D-sized vs EW-U100 by layer; turnover and costs",
               "books": books, "active_total": total, "decomposition": decomp, "text_names": names,
               "registered_reference": {"pd1_rows": {k: {"sharpe": pd1["rows"][k]["sharpe"]["sharpe"],
                                                         "turnover_one_way": pd1["rows"][k]["turnover_one_way"],
                                                         "ann_excess_return": pd1["rows"][k]["ann_excess_return"]}
                                                     for k in ("sized", "sel_ew", "ew_u100")},
                                        "source": "results/analysis/pd1.json (registered PD1; repeated)"},
               "figure_data": {"weekly_layers_csv": C.rel(FIGDIR / "e4_weekly_layers.csv", root)},
               "consistency": {"books_equal_logged_returns_max_abs_diff": check, "n_learned_weeks": len(learned)},
               "settings": {"cost_bps": P1.COST_BPS, "nw_lags": S.NW_LAGS, "weeks": "learned weeks only",
                            "runs": {"primary": C.rel(prim.path, root), "V-J0": C.rel(j0.path, root)}}}
    inputs = {"primary weights": prim.file("weights.parquet"), "primary returns": prim.file("daily_returns.parquet"),
              "V-J0 weights": j0.file("weights.parquet"), "V-J0 returns": j0.file("daily_returns.parquet"),
              "pd1": root / "results/analysis/pd1.json"}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="layer attribution of the system book specified after the PD1 outcome")
    print(f"active total {total['net_bps_per_week']['mean']:.2f} bps/wk CI {total['net_bps_per_week']['ci']}")
    for o, d in decomp.items():
        print(o, {k: round(v['net_bps_per_week']['mean'], 2) for k, v in d.items() if 'net_bps_per_week' in v})
    print(f"text entrants - displaced {names['entrants_minus_displaced_bps']['mean']:.2f} bps/wk")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
