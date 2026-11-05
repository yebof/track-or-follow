#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E2: "valid but already priced". Where in time do the typed news judgments'
return correlations sit, relative to the fill the system actually trades?

    .venv/bin/python experiments/exploratory/e2_priced_horizon.py    # -> results/exploratory/e2_priced_horizon.json
                                                                     #    + results/exploratory/figdata/e2_*.csv

Why exploratory. H2 (prereg §3.2) registered one window, AR[0,+1] (plus the placebo AR[-5,-1]), and H3 (§3.3) the
weekly rank-IC of the combined logit. The horizon profile, the decision-time alignment, the timing shares and the
stock-week N3 rows below were specified after both outcomes were known (H2 stated, H3 fallback) to ask how the two fit
together. No p-value is computed; each statistic is a mean of weekly Spearman correlations with a two-sided
Newey-West(4) 95% interval (stats.nw_test), as H2's own statistic, and decides nothing.

Item level (the registered H2 frame, rebuilt with experiments/analysis/h2_validity.py's `item_rows` / `c2_frame`
read-only, from the D1 start; its AR[0,+1] weekly mean must reproduce h2.json's 0.1192 exactly, checked): E[effect]
of every C2-eligible item vs
- abnormal returns (learn.labels.abnormal_returns: sum of r_i - beta_i r_SPY, close to close, beta the trailing 252
  sessions ending before the window) over windows of trading days relative to the item's event session d0:
  [-5,-1] (the registered placebo), [0,+1] (the registered H2 label), [+2,+5], [+6,+10], and every single day
  -5..+10 (the figure's event-time profile);
- the same relative to the decision session d_t (the item's week's decision: d0 <= d_t because items are created
  before the 16:00 ET cutoff c_t): single days -4..+10 (the figure's decision-time profile) and AR[d0, d_t], the part
  of the reaction that closes before the fill;
- the overnight gap from close(d_t) to open(e_t) (the fill), stock minus the U_t median;
- the next weekly holding-period return the system trades: r - m_t, r = open(e_{t+1}) / open(e_t) - 1 (learn.labels,
  the T1 label), and the week after it (t+1).
Timing shares: items whose AR[0,+1] window ends at or before d_t (the whole registered reaction happens before the
next open), items with d0 = d_t (day +1 spans the open), counted over the frame.

Stock-week level (N3, one row per covered stock-week of the primary cell's learned weeks): f_net = (E[net_effect] - 2)
/ 2 from MAIN's Layer-A features, and, for comparison, the week's mean item E[effect] and the primary cell's logged
news-agent contribution c_news, vs AR over decision-relative windows [-4,0] (the news week), single days -4..+10, the
overnight gap, the traded r - m_t and the next week's r - m_{t+1}.
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
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

NAME = "e2_priced_horizon"
GEN = "main_emu_wclean_u100"
ITEM_WINDOWS = {"placebo_m5_m1": (-5, -1), "ar_0_1": (0, 1), "ar_2_5": (2, 5), "ar_6_10": (6, 10)}
EVENT_DAYS = tuple(range(-5, 11))
DECISION_DAYS = tuple(range(-4, 11))
MIN_UNITS = H2.MIN_UNITS
FIGDIR = X.OUT_DIR / "figdata"


def weekly(f: pd.DataFrame, x: str, y: str) -> dict:
    ws = S.weekly_spearman(f, x, y, "t", MIN_UNITS)
    s = ws.dropna(subset=["rho"])
    t = S.nw_test(s["rho"].to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean": t["mean"], "ci": t["ci"], "se": t["se"], "n_weeks": int(len(s)),
            "n_weeks_undefined": int(ws["rho"].isna().sum()), "n_units": int(s["n"].sum())}


def decision_ar(book, keys: pd.DataFrame, window: tuple[int, int]) -> np.ndarray:
    """AR over sessions d_t + window (abnormal_returns with d0 = d_t: a created_at at 12:00 ET on d_t)."""
    from jevtrader.learn.labels import abnormal_returns

    ev = pd.DataFrame({"symbol": keys["symbol"].to_numpy(),
                       "created_at": [pd.Timestamp(f"{pd.Timestamp(t).date()} 12:00").tz_localize(C.ET)
                                      for t in keys["t"]]})
    return abnormal_returns(book, ev, window=window)["ar"].to_numpy(dtype=float)


def overnight_gap(book, keys: pd.DataFrame, universe: dict) -> pd.Series:
    """open(e_t) / close(d_t) - 1, minus its median over U_t (as the label's r - m_t); indexed like `keys`."""
    ses = book.sessions
    out = {}
    for t, g in keys.groupby("t"):
        k = ses.searchsorted(pd.Timestamp(t))
        if k + 1 >= len(ses) or ses[k] != pd.Timestamp(t):
            continue
        e = ses[k + 1]
        names = [s for s in universe.get(pd.Timestamp(t), ()) if s in book.open.columns]
        gap = book.open.loc[e, names] / book.close.loc[ses[k], names] - 1.0
        gap = gap - float(np.nanmedian(gap.to_numpy(dtype=float)))
        for i, s in zip(g.index, g["symbol"]):
            out[i] = float(gap.get(s, np.nan))
    return pd.Series(out, dtype=float)


def main(argv=None) -> int:
    from jevtrader.learn.labels import abnormal_returns

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window("W_clean", "E2")
    d1 = C.d1_start(root)
    gen = root / "results/judgments" / GEN
    items = H2.item_rows(gen, start, end)
    frame = H2.c2_frame(items)
    frame = frame[frame["t"] >= pd.Timestamp(d1["start"])].reset_index(drop=True)
    guard.check(frame["t"].unique(), "E2 labels")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    book, ses = px.book, px.book.sessions
    ev = frame[["symbol", "created_at"]]
    for k, w in ITEM_WINDOWS.items():
        r = abnormal_returns(book, ev, window=w)
        frame[k] = r["ar"].to_numpy(dtype=float)
        if k == "ar_0_1":
            frame["d0"] = pd.to_datetime(r["d0"])
    for d in EVENT_DAYS:
        frame[f"ev_{d:+d}"] = abnormal_returns(book, ev, window=(d, d))["ar"].to_numpy(dtype=float)
    # timing of each item relative to its decision session d_t
    k_t = ses.searchsorted(frame["t"].to_numpy())
    k_0 = ses.searchsorted(frame["d0"].to_numpy())
    frame["pos"] = k_0 - k_t  # <= 0 for items created before the 16:00 ET cutoff of d_t
    frame["pre_fill_ar"] = np.nan  # AR[d0, d_t]: the part of the reaction that closes before the fill
    for p, g in frame.groupby("pos"):
        if p <= 0:
            frame.loc[g.index, "pre_fill_ar"] = abnormal_returns(book, g[["symbol", "created_at"]],
                                                                 window=(0, int(-p)))["ar"].to_numpy(dtype=float)
    keys = frame[["t", "symbol"]]
    for d in DECISION_DAYS:
        frame[f"dt_{d:+d}"] = decision_ar(book, keys, (d, d))
    frame["gap_rel"] = overnight_gap(book, keys, px.universe).reindex(frame.index)
    weeks = sorted(frame["t"].unique())
    cal = [w for w in px.dates if w >= weeks[0]]
    labels = C.load_labels(px, cal)
    lab = labels.table[["t", "symbol", "r", "m"]].assign(rel=lambda x: x["r"] - x["m"])
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    lab_next = lab[["t", "symbol", "rel"]].rename(columns={"rel": "rel_next"}).assign(
        t=lambda x: x["t"].map({v: k for k, v in nxt.items()}))
    frame = frame.merge(lab[["t", "symbol", "rel"]], on=["t", "symbol"], how="left").merge(
        lab_next.dropna(subset=["t"]), on=["t", "symbol"], how="left")

    # consistency: the registered H2 statistic on this frame
    reg = C.read_json(root / "results/analysis/h2.json")["confirmatory"]
    mine = H2.weekly_test(frame.rename(columns={"ar_0_1": "ar"}))
    consistency = {"h2_mean_registered": reg["mean"], "h2_mean_recomputed": mine["mean"],
                   "abs_diff": abs(reg["mean"] - mine["mean"]), "n_units": len(frame),
                   "n_units_registered": reg.get("n_units"), "items_after_cutoff": int((frame["pos"] > 0).sum())}
    if consistency["abs_diff"] > 1e-12 or consistency["items_after_cutoff"]:
        raise SystemExit(f"E2: frame does not reproduce the registered H2 statistic: {consistency}")

    item = {"windows_event_time": {k: weekly(frame, "E_effect", k) for k in ITEM_WINDOWS},
            "pre_fill_ar_d0_to_dt": weekly(frame, "E_effect", "pre_fill_ar"),
            "overnight_gap_to_fill": weekly(frame, "E_effect", "gap_rel"),
            "traded_holding_week_r_minus_m": weekly(frame, "E_effect", "rel"),
            "next_holding_week_r_minus_m": weekly(frame, "E_effect", "rel_next")}
    item["event_day_profile"] = {f"{d:+d}": weekly(frame, "E_effect", f"ev_{d:+d}") for d in EVENT_DAYS}
    item["decision_day_profile"] = {f"{d:+d}": weekly(frame, "E_effect", f"dt_{d:+d}") for d in DECISION_DAYS}
    n = len(frame)
    timing = {"n_items": n,
              "share_window_0_1_closed_before_fill": float((frame["pos"] <= -1).mean()),
              "share_d0_on_decision_day": float((frame["pos"] == 0).mean()),
              "share_d0_before_decision_day": float((frame["pos"] < 0).mean()),
              "pos_distribution": {str(int(k)): int(v) for k, v in frame["pos"].value_counts().sort_index().items()},
              "median_sessions_d0_to_fill": float((1 - frame["pos"]).median()),
              "definition": "pos = session index of d0 minus that of d_t (<= 0); the registered AR[0,+1] window "
                            "closes before open(e_t) iff pos <= -1; day +1 spans the open iff pos = 0"}
    by_timing = {}
    for tag, g in (("window_closed_before_fill", frame[frame["pos"] <= -1]),
                   ("d0_on_decision_day", frame[frame["pos"] == 0])):
        by_timing[tag] = {"n_items": int(len(g)), "ar_0_1": weekly(g, "E_effect", "ar_0_1"),
                          "traded_holding_week": weekly(g, "E_effect", "rel"),
                          "ar_2_5": weekly(g, "E_effect", "ar_2_5")}

    # stock-week level: N3 f_net, the week's mean item E[effect] and the primary cell's news contribution
    feats = pd.read_parquet(gen / "features.parquet")
    feats["t"] = pd.to_datetime(feats["t"])
    fnet = feats[(feats["feature"].astype(str) == "f_net") & feats["covered"].astype(bool)][
        ["t", "symbol", "value"]].rename(columns={"value": "f_net"})
    fnet = fnet[np.isfinite(fnet["f_net"].to_numpy(dtype=float))]
    prim = C.read_eval_run(C.find_eval_dir(root, "W_clean", "primary", GEN), guard, "E2 primary")
    pr = prim.learned[["t", "symbol", "c_news", "L_W_bar"]]
    sw = pr.merge(fnet, on=["t", "symbol"], how="left").merge(
        frame.groupby(["t", "symbol"])["E_effect"].mean().rename("item_mean_E").reset_index(), on=["t", "symbol"],
        how="left")
    for d in DECISION_DAYS:
        sw[f"dt_{d:+d}"] = decision_ar(book, sw[["t", "symbol"]], (d, d))
    sw["news_week_ar_m4_0"] = decision_ar(book, sw[["t", "symbol"]], (-4, 0))
    sw["ar_p6_p10"] = decision_ar(book, sw[["t", "symbol"]], (6, 10))
    sw["gap_rel"] = overnight_gap(book, sw[["t", "symbol"]], px.universe).reindex(sw.index)
    sw = sw.merge(lab[["t", "symbol", "rel"]], on=["t", "symbol"], how="left").merge(
        lab_next.dropna(subset=["t"]), on=["t", "symbol"], how="left")
    stock = {}
    for x in ("f_net", "item_mean_E", "c_news"):
        g = sw.dropna(subset=[x]) if x != "c_news" else sw
        stock[x] = {"n_stock_weeks": int(len(g)),
                    "news_week_ar_m4_0": weekly(g, x, "news_week_ar_m4_0"),
                    "overnight_gap_to_fill": weekly(g, x, "gap_rel"),
                    "traded_holding_week_r_minus_m": weekly(g, x, "rel"),
                    "next_holding_week_r_minus_m": weekly(g, x, "rel_next"),
                    "ar_p6_p10_decision_time": weekly(g, x, "ar_p6_p10"),
                    "decision_day_profile": {f"{d:+d}": weekly(g, x, f"dt_{d:+d}") for d in DECISION_DAYS}}

    # figure-ready tables (exploratory; for the paper figure on E2)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for level, prof in (("item E[effect]", item["event_day_profile"]),):
        for d, v in prof.items():
            rows.append({"panel": "event_time", "series": level, "day": int(d), **_flat(v)})
    for d, v in item["decision_day_profile"].items():
        rows.append({"panel": "decision_time", "series": "item E[effect]", "day": int(d), **_flat(v)})
    for d, v in stock["f_net"]["decision_day_profile"].items():
        rows.append({"panel": "decision_time", "series": "N3 f_net", "day": int(d), **_flat(v)})
    prof = pd.DataFrame(rows)
    bars = []
    for series, block in (("item E[effect]", {"AR[-5,-1]": item["windows_event_time"]["placebo_m5_m1"],
                                              "AR[0,+1]": item["windows_event_time"]["ar_0_1"],
                                              "AR[+2,+5]": item["windows_event_time"]["ar_2_5"],
                                              "AR[+6,+10]": item["windows_event_time"]["ar_6_10"],
                                              "AR[d0,d_t] (before fill)": item["pre_fill_ar_d0_to_dt"],
                                              "gap close(d_t)->open(e_t)": item["overnight_gap_to_fill"],
                                              "traded week r-m (open e_t->open e_t+1)":
                                                  item["traded_holding_week_r_minus_m"],
                                              "next week r-m": item["next_holding_week_r_minus_m"]}),
                          ("N3 f_net", {"AR[d_t-4,d_t] (news week)": stock["f_net"]["news_week_ar_m4_0"],
                                        "gap close(d_t)->open(e_t)": stock["f_net"]["overnight_gap_to_fill"],
                                        "traded week r-m (open e_t->open e_t+1)":
                                            stock["f_net"]["traded_holding_week_r_minus_m"],
                                        "next week r-m": stock["f_net"]["next_holding_week_r_minus_m"],
                                        "AR[d_t+6,d_t+10]": stock["f_net"]["ar_p6_p10_decision_time"]})):
        for k, v in block.items():
            bars.append({"series": series, "window": k, **_flat(v)})
    bars = pd.DataFrame(bars)
    for df, nm in ((prof, "e2_profile.csv"), (bars, "e2_windows.csv")):
        with open(FIGDIR / nm, "w") as fh:
            fh.write(f"# {X.LABEL}; generated by experiments/exploratory/e2_priced_horizon.py; mean weekly Spearman "
                     f"with a two-sided NW(4) 95% CI\n")
            df.to_csv(fh, index=False)
    payload = {"question": "valid but already priced: the horizon profile of typed news judgments vs the traded "
                           "holding-period return",
               "frame": {"n_items": n, "n_weeks": len(weeks), "start": str(weeks[0].date()),
                         "end": str(weeks[-1].date()), "rule": "the registered H2 frame (h2_validity.c2_frame)"},
               "item_level": item, "timing": timing, "item_level_by_timing": by_timing, "stock_week_level": stock,
               "registered_reference": {"h2_confirmatory_mean": reg["mean"], "h2_ci": reg["ci"],
                                        "h2_placebo": C.read_json(root / "results/analysis/h2.json")[
                                            "placebo_ar_m5_m1"]["mean"],
                                        "h3_primary_ic": C.read_json(root / "results/analysis/h3.json")[
                                            "confirmatory"]["mean_a"],
                                        "source": "results/analysis/h2.json, h3.json (repeated, not recomputed)"},
               "figure_data": {"profile_csv": C.rel(FIGDIR / "e2_profile.csv", root),
                               "windows_csv": C.rel(FIGDIR / "e2_windows.csv", root),
                               "profile": prof.to_dict("records"), "windows": bars.to_dict("records")},
               "consistency": consistency,
               "settings": {"nw_lags": S.NW_LAGS, "min_units_per_week": MIN_UNITS, "ci": "two-sided NW 95%",
                            "price_cap": px.cap, "d1_start": str(pd.Timestamp(d1["start"]).date())}}
    inputs = {"work.jsonl": gen / "memory" / "work.jsonl", "MAIN features": gen / "features.parquet",
              "primary predictions": prim.file("predictions.parquet"), "h2": root / "results/analysis/h2.json",
              "universe": px.root / C.UNIVERSE_FILE}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="horizon profile and timing decomposition specified after the H2 and H3 outcomes")
    w = item["windows_event_time"]
    print(f"items {n}: AR[0,1] {w['ar_0_1']['mean']:.4f}  AR[2,5] {w['ar_2_5']['mean']:.4f}  AR[6,10] "
          f"{w['ar_6_10']['mean']:.4f}  traded {item['traded_holding_week_r_minus_m']['mean']:.4f}  "
          f"closed-before-fill share {timing['share_window_0_1_closed_before_fill']:.3f}")
    print(f"f_net: news week {stock['f_net']['news_week_ar_m4_0']['mean']:.4f} traded "
          f"{stock['f_net']['traded_holding_week_r_minus_m']['mean']:.4f}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


def _flat(v: dict) -> dict:
    return {"mean": v["mean"], "ci_low": v["ci"][0], "ci_high": v["ci"][1], "n_weeks": v["n_weeks"],
            "n_units": v["n_units"]}


if __name__ == "__main__":
    sys.exit(main())
