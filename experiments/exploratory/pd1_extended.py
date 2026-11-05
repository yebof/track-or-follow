#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the extended portfolio comparison around PD1 (Table 4x, the stress table and the
TA-lite / Mono-CoT subset comparison).

    .venv/bin/python experiments/exploratory/pd1_extended.py      # -> results/exploratory/pd1_extended.json

Why exploratory. The registered PD1 (prereg §4.1, §4.4; experiments/analysis/pd1_portfolio.py -> results/analysis/
pd1.json, Table 4) has at most eight rows and fixed columns. Everything here beyond those rows and columns was
specified after the unlock: the extra rows (reversal, low-volatility, B8 logistic, the LM-dictionary reader's books,
Random-20 as a row), the extra columns (CAGR, Calmar, cost drag, IR vs EW-U100 with an interval, the Fama-French
five-factor + momentum attribution), the stress table (design_v2_delta PE-6: a registered design item, but not in the
pre-registration) and the S30 x 60-week comparison with the TA-lite and Mono-CoT books (prereg §6 PE-11 registers the
restriction rule, not these books). The registered numbers themselves (DSR over M = 73, Random-20 percentile) are
read from results/analysis/pd1.json and only repeated.

Conventions (the registered PD1's, imported read-only from pd1_portfolio.py): the primary cell's decision dates,
filled at the next open, 15 bps per side, cash at DFF, held to the primary cell's holding end; Sharpe of the daily
excess return with the studentized circular block-bootstrap 95% interval (block 5, B = 4,999, seed 0).

Rows (U100 x W_clean): D-sized and D-sel-EW (primary), EW-U100, SPY, vol-managed EW-U100, momentum 12-1, short-term
reversal (top 20 by the code feature rev_1w, i.e. the 1-week losers), low volatility (the 20 lowest vol_60d), B8
logistic and B9 LightGBM on the 14 code features (P-clean, cold start at the window start, baselines.ml), the LM
dictionary in the untrusted slot (V-reader-LM, both books, registered trial runs), the code-only system (V-codeonly
D-sized); Random-20 (median and 5-95% of 1,000 seeded draws, from pd1.json).
Columns: CAGR; annualized excess return; volatility; Sharpe [CI]; maximum drawdown; Calmar = CAGR / |MDD|; one-way
turnover; cost drag = annualized mean daily return at 0 bps minus at 15 bps; IR vs EW-U100 = annualized mean / sd of
the daily active return, with the same studentized block-bootstrap interval; FF5 + momentum alpha [NW(5) CI] and
market beta (eval.attribution.french_attribution, on the dates the French data cover); own-factor alpha (PE-3).

Stress (PE-6 rule): the named episodes documented in literature/summaries only, the April 2-9, 2025 tariff shock
[li2025time, p.5] and the October 10, 2025 tariff shock [fan2025aitrader, p.10-11], as cumulative returns over those
sessions; and the mechanical stress weeks (holding weeks with SPY in the bottom decile of the W_clean holding weeks,
or VIXCLS > 30 on any session of the week), listed by date only. No other episode is named, shaded or annotated.

Subset (PE-11): TA-lite and Mono-CoT ran on S30 x the last 60 W_clean decisions. On exactly those stock-weeks:
TA-lite's native book (EW over BUY names, <= 20; baselines.llm.native_book's rule), Mono-CoT's p > 0.5 book (EW over
names with p_outperform > 0.5, <= 20 by p; Mono-CoT has no BUY/SELL output, so this is our reading of "native"), the
typed system's signal book restricted to those names (D-sel-EW names in S30, EW), and a common rule for all three
scores (EW top 6 = 20% of the 30 names by TA-lite p, Mono-CoT p, typed L); EW-S30 is the benchmark. Weekly rank-IC of
each score with r - m over the same stock-weeks (NW(4) 95% CI). Estimates and intervals only: no test.
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

NAME = "pd1_extended"
GEN = "main_emu_wclean_u100"
TAL = "results/gens/tal_ta-lite_wclean-last60_s30"
MCOT = "results/judgments/mcot_mono-cot_wclean-last60_s30"
EPISODES = {"tariff_2025_04": ("2025-04-02", "2025-04-09", "April 2-9, 2025 tariff shock [li2025time, p.5]"),
            "tariff_2025_10": ("2025-10-10", "2025-10-10", "October 10, 2025 tariff shock [fan2025aitrader, p.10-11]")}
TOP_SUBSET = 6  # 20% of the 30 subset names (K / |U100| = 20 / 100)


def vix_series(root: Path) -> pd.Series:
    import json

    obs = json.loads((root / "data/raw/macro/fred/VIXCLS.json").read_text())["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")})
    return s.sort_index()


def metrics(res, res0, rf: pd.Series, ew, french, window_end, fac, n_boot: int) -> dict:
    from jevtrader.eval import metrics as M
    from jevtrader.eval.attribution import french_attribution, style_attribution

    r = res.daily_returns
    rfr = rf.reindex(r.index).fillna(0.0)
    ex = r - rfr
    nav = (1 + r.fillna(0.0)).cumprod()
    years = len(r) / 252.0
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else None
    mdd = float(M.max_drawdown(nav))
    out = {"n_days": len(r), "cagr": cagr, "ann_excess_return": float(ex.mean() * 252),
           "ann_vol": float(ex.std(ddof=1) * np.sqrt(252)),
           "sharpe": S.sharpe_ci(ex, P1.SHARPE_BLOCK, n_boot, P1.SEED), "max_drawdown": mdd,
           "calmar": (cagr / abs(mdd)) if cagr is not None and mdd < 0 else None,
           "turnover_one_way": float(0.5 * res.turnover.mean()) if len(res.turnover) else 0.0,
           "cost_drag_ann": float((res0.daily_returns - r).mean() * 252)}
    if ew is not None:
        act = (r - ew.daily_returns.reindex(r.index)).dropna()
        out["ir_vs_ew"] = S.sharpe_ci(act, P1.SHARPE_BLOCK, n_boot, P1.SEED)
    if french is not None:
        fa = french_attribution(r, rf, french, window_end=window_end)
        d = fa.get("daily", fa)
        coef = d.get("coef", {}) if isinstance(d, dict) else {}
        out["ff5_mom"] = {"alpha_annualized": d.get("alpha_annualized"), "ci": d.get("alpha_ci95_annualized"),
                          "beta_mkt": coef.get("Mkt-RF"), "n": d.get("n"), "coverage": d.get("coverage")}
    if fac is not None:
        att = style_attribution(r, rf, fac).get("daily", {})
        out["own_factor_alpha"] = {"annualized": att.get("alpha_annualized"), "ci": att.get("alpha_ci95_annualized")}
    return out


def episode_returns(results: dict, lo: str, hi: str) -> dict:
    out = {}
    for k, res in results.items():
        r = res.daily_returns
        w = r[(r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))]
        out[k] = {"cum_return": float((1 + w).prod() - 1) if len(w) else None, "n_sessions": len(w)}
    return out


def holding_weeks(dates, hold_end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ends = list(dates[1:]) + [pd.Timestamp(hold_end)]
    return list(zip(dates, ends))


def mechanical_weeks(results: dict, dates, hold_end, vix: pd.Series) -> dict:
    spy = results["spy"].daily_returns
    wk = []
    for t, e in holding_weeks(dates, hold_end):
        m = (spy.index > t) & (spy.index <= e)
        v = vix[(vix.index > t) & (vix.index <= e)]
        wk.append({"decision": t, "spy": float((1 + spy[m]).prod() - 1), "vix_max": float(v.max()) if len(v) else None})
    w = pd.DataFrame(wk)
    cut = w["spy"].quantile(0.10)
    sel = w[(w["spy"] <= cut) | (w["vix_max"].fillna(0) > 30)]
    rows = {}
    for k, res in results.items():
        r = res.daily_returns
        vals = []
        for t in sel["decision"]:
            e = dict(holding_weeks(dates, hold_end))[t]
            m = (r.index > t) & (r.index <= e)
            vals.append(float((1 + r[m]).prod() - 1))
        rows[k] = {"mean_weekly_return": float(np.mean(vals)) if vals else None,
                   "sum_return": float(np.sum(vals)) if vals else None}
    return {"rule": "SPY holding-week return in the bottom decile of W_clean's holding weeks, or VIXCLS > 30 on any "
                    "session of the week (listed by date only; PE-6)",
            "spy_bottom_decile_cut": float(cut), "n_weeks": int(len(sel)),
            "weeks": [{"decision": str(pd.Timestamp(x.decision).date()), "spy": x.spy, "vix_max": x.vix_max}
                      for x in sel.itertuples()],
            "books": rows}


def subset_books(root: Path, prim, px, rf, hold_end, ma, lab) -> dict:
    from jevtrader.baselines.classic import top_k_scores
    from jevtrader.decide.select import tiebreak_key

    tal = pd.read_parquet(root / TAL / "answers.parquet")
    mc = pd.read_parquet(root / MCOT / "answers.parquet")
    for d in (tal, mc):
        d["t"] = pd.to_datetime(d["t"])
    dates = sorted(tal["t"].unique())
    names = {t: sorted(g["symbol"]) for t, g in tal.groupby("t")}
    s30 = {t: sorted(C.subset(list(px.universe[t]), "S30", root)) for t in dates}
    same = all(set(names[t]) == set(s30[t]) for t in dates)
    mc_names = {t: sorted(g["symbol"]) for t, g in mc.groupby("t")}
    same_mc = all(set(mc_names.get(t, [])) == set(names[t]) for t in dates)
    pred = prim.predictions[prim.predictions["t"].isin(dates)]
    w = pd.read_parquet(prim.file("weights.parquet"))
    w["t"] = pd.to_datetime(w["t"])
    sel = w[(w["book"] == "sel_ew") & (w["t"].isin(dates))]

    def ranked(scores: dict[str, float], k: int) -> dict[str, float]:
        s = sorted(((sym, p) for sym, p in scores.items() if p is not None and np.isfinite(p)),
                   key=lambda sp: (-sp[1], tiebreak_key(sp[0])))
        top = [x for x, _ in s[:k]]
        return {x: 1.0 / len(top) for x in top} if top else {}

    tg: dict[str, dict] = {k: {} for k in ("ew_s30", "tal_native", "mcot_p50", "typed_sel_s30", "tal_top6",
                                           "mcot_top6", "typed_top6")}
    for t in dates:
        nm = names[t]
        tg["ew_s30"][t] = {s: 1.0 / len(nm) for s in nm}
        a = tal[tal["t"] == t]
        buys = {r.symbol: (r.p_outperform if pd.notna(r.p_outperform) else -1.0) for r in a.itertuples()
                if r.decision == "BUY"}
        tg["tal_native"][t] = ranked(buys, 20)
        m = mc[mc["t"] == t]
        tg["mcot_p50"][t] = ranked({r.symbol: r.p_outperform for r in m.itertuples()
                                    if pd.notna(r.p_outperform) and r.p_outperform > 0.5}, 20)
        sw = [s for s in sel.loc[sel["t"] == t, "symbol"] if s in set(nm)]
        tg["typed_sel_s30"][t] = {s: 1.0 / len(sw) for s in sw} if sw else {}
        tg["tal_top6"][t] = ranked({r.symbol: r.p_outperform for r in a.itertuples()}, TOP_SUBSET)
        tg["mcot_top6"][t] = ranked({r.symbol: r.p_outperform for r in m.itertuples()}, TOP_SUBSET)
        p = pred[(pred["t"] == t) & (pred["symbol"].isin(nm))]
        tg["typed_top6"][t] = top_k_scores(pd.Series(p["L"].to_numpy(), index=p["symbol"]), TOP_SUBSET)
    res = {k: P1.backtest(px, v, rf, hold_end, ma) for k, v in tg.items()}
    res0 = {k: P1.backtest(px, v, rf, hold_end, ma, cost=0.0) for k, v in tg.items()}
    empty = {k: int(sum(1 for t in dates if not v[t])) for k, v in tg.items()}
    # weekly rank-IC of each score on the same stock-weeks
    sc = tal[["t", "symbol", "p_outperform"]].rename(columns={"p_outperform": "tal"}).merge(
        mc[["t", "symbol", "p_outperform"]].rename(columns={"p_outperform": "mcot"}), on=["t", "symbol"], how="outer"
    ).merge(pred[["t", "symbol", "L"]].rename(columns={"L": "typed"}), on=["t", "symbol"], how="left").merge(
        lab, on=["t", "symbol"], how="left")
    ic = {}
    for col in ("tal", "mcot", "typed"):
        wk = {}
        for t, g in sc.groupby("t"):
            g = g.dropna(subset=[col, "rel"])
            if len(g) >= 5 and g[col].nunique() > 1:
                wk[t] = g[col].corr(g["rel"], method="spearman")
        v = pd.Series(wk, dtype=float)
        tt = S.nw_test(v.to_numpy(), S.NW_LAGS, "two-sided")
        ic[col] = {"mean": tt["mean"], "ci": tt["ci"], "n_weeks": int(len(v)), "n_dropped_weeks": len(dates) - len(v)}
    return {"dates": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())], "n_weeks": len(dates),
            "tal_names_equal_S30": bool(same), "mcot_names_equal_tal": bool(same_mc), "empty_weeks": empty,
            "_res": res, "_res0": res0, "ic": ic}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", default="W_clean")
    ap.add_argument("--sharpe-boot", type=int, default=P1.SHARPE_BOOT)
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "PD1 exploratory")
    prim = C.read_eval_run(C.find_eval_dir(root, a.window, "primary", GEN), guard, "PD1x primary")
    code = C.read_eval_run(C.find_eval_dir(root, a.window, "V-codeonly", GEN), guard, "PD1x V-codeonly")
    lmr = C.read_eval_run(C.find_eval_dir(root, a.window, "V-reader-LM", GEN), guard, "PD1x V-reader-LM")
    gen = root / "results/judgments" / GEN
    dates = prim.dates
    hold_end = pd.Timestamp(prim.metrics["hold_end"])
    px = C.load_prices(guard, start, end, pad_before=P1.FULL_HISTORY)
    from jevtrader.baselines.classic import top_k_scores
    from jevtrader.baselines.ml import LearnedBaseline, standardized_panel
    from jevtrader.data.fundamentals import CODE_FEATURES as F_CODE
    from jevtrader.eval.attribution import FACTORS_DIR, factor_inputs, load_french, own_factor_returns
    from jevtrader.features.technical import CODE_FEATURES as T_CODE
    from jevtrader.pipeline import inputs as I

    universe = {d: px.universe[d] for d in dates}
    ma = I.ma_exit_set(px.root, sorted({s for v in universe.values() for s in v}))
    rf = C.cash_rate(px)
    rf = rf if rf is not None else pd.Series(0.0, index=px.book.sessions)
    labels = C.load_labels(px, dates, universe)
    lab = labels.table[["t", "symbol", "r", "m"]].assign(rel=lambda x: x["r"] - x["m"])[["t", "symbol", "rel"]]
    code_raw = pd.read_parquet(gen / "code_features.parquet")
    code_raw["t"] = pd.to_datetime(code_raw["t"])
    firm = I.load_firm_info(gen)
    fac = own_factor_returns(px.book, factor_inputs(code_raw, firm), universe, dates, end=hold_end,
                             cash_rate_daily=rf, ma_exits=ma)
    fdir = root / FACTORS_DIR
    french = load_french(fdir) if fdir.exists() else None

    def feat(name):
        return code_raw[code_raw["feature"] == name].pivot_table(index="t", columns="symbol", values="value",
                                                                  aggfunc="first")

    rev, vol = feat("rev_1w"), feat("vol_60d")
    tg = {**{b: t for b, t in P1.targets_from_weights(prim).items() if b in ("sized", "sel_ew")},
          **P1.baseline_targets(px, dates, universe, code_raw, labels),
          "codeonly": P1.targets_from_weights(code)["sized"],
          "lm_sized": P1.targets_from_weights(lmr)["sized"], "lm_sel_ew": P1.targets_from_weights(lmr)["sel_ew"]}
    tg["reversal"] = {d: top_k_scores(rev.loc[d].reindex(list(universe[d])) if d in rev.index else pd.Series(
        dtype=float), P1.K_BOOK) for d in dates}
    tg["low_vol"] = {d: top_k_scores(-vol.loc[d].reindex(list(universe[d])) if d in vol.index else pd.Series(
        dtype=float), P1.K_BOOK) for d in dates}
    codef = tuple(T_CODE) + tuple(F_CODE)
    wide = {d: code_raw[code_raw["t"] == d].pivot_table(index="symbol", columns="feature", values="value",
                                                          aggfunc="first").reindex(list(universe[d])) for d in dates}
    Xp = standardized_panel(wide, codef)
    tg["logistic"], _ = LearnedBaseline("logit", "P-clean", pd.Timestamp(dates[0]), codef, k=P1.K_BOOK,
                                        seed=P1.SEED).run(Xp, labels, list(dates), {d: list(universe[d]) for d in dates})
    order = ("sized", "sel_ew", "ew_u100", "spy", "vm_ew_u100", "momentum_12_1", "reversal", "low_vol", "logistic",
             "lightgbm", "lm_sized", "lm_sel_ew", "codeonly")
    labels_ = {"sized": "System book (D-sized)", "sel_ew": "Signal book (D-sel-EW)", "ew_u100": "EW-U100",
               "spy": "SPY", "vm_ew_u100": "Vol-managed EW-U100", "momentum_12_1": "Momentum 12-1",
               "reversal": "Short-term reversal", "low_vol": "Low volatility", "logistic": "Logistic (B8, code)",
               "lightgbm": "LightGBM (B9, code)", "lm_sized": "LM dictionary reader, D-sized",
               "lm_sel_ew": "LM dictionary reader, D-sel-EW", "codeonly": "Code-only system"}
    res = {k: P1.backtest(px, tg[k], rf, hold_end, ma) for k in order}
    res0 = {k: P1.backtest(px, tg[k], rf, hold_end, ma, cost=0.0) for k in order}
    logged = pd.read_parquet(prim.file("daily_returns.parquet"))
    check = {b: float((res[b].daily_returns - logged[b].reindex(res[b].daily_returns.index)).abs().max())
             for b in ("sized", "sel_ew", "ew_u100")}
    rows = {k: {"label": labels_[k], **metrics(res[k], res0[k], rf, res["ew_u100"] if k != "ew_u100" else None,
                                                french, end, fac, a.sharpe_boot)} for k in order}
    reg = C.read_json(root / "results/analysis/pd1.json") if (root / "results/analysis/pd1.json").exists() else None
    rnd = (reg or {}).get("random20")
    stress = {"named": {k: {"label": lab_, "from": lo, "to": hi, "books": episode_returns(res, lo, hi)}
                        for k, (lo, hi, lab_) in EPISODES.items()},
              "mechanical": mechanical_weeks(res, dates, hold_end, vix_series(root))}
    sub = subset_books(root, prim, px, rf, hold_end, ma, lab)
    sres, sres0 = sub.pop("_res"), sub.pop("_res0")
    sub_labels = {"ew_s30": "EW-S30 (benchmark)", "tal_native": "TA-lite native (BUY, EW)",
                  "mcot_p50": "Mono-CoT p > 0.5 (EW)", "typed_sel_s30": "Typed D-sel-EW names in S30 (EW)",
                  "tal_top6": "TA-lite top 6 by p", "mcot_top6": "Mono-CoT top 6 by p", "typed_top6": "Typed top 6 by L"}
    sub["rows"] = {k: {"label": sub_labels[k], **metrics(sres[k], sres0[k], rf, sres["ew_s30"] if k != "ew_s30"
                                                         else None, None, end, None, a.sharpe_boot)}
                   for k in sub_labels}
    sub["order"] = list(sub_labels)
    payload = {"rows": rows, "order": list(order), "window_dates": [str(dates[0].date()), str(dates[-1].date())],
               "hold_end": str(hold_end.date()), "n_decisions": len(dates),
               "registered_reference": {"source": "results/analysis/pd1.json (registered PD1; repeated, not recomputed)",
                                        "dsr": (reg or {}).get("dsr", {}).get("by_book"),
                                        "n_trials": (reg or {}).get("dsr", {}).get("n_trials"),
                                        "random20": rnd},
               "consistency": {"system_books_equal_logged_returns_max_abs_diff": check},
               "stress": stress, "subset_s30_last60": sub,
               "settings": {"cost_bps": P1.COST_BPS, "sharpe_block": P1.SHARPE_BLOCK, "sharpe_boot": a.sharpe_boot,
                            "seed": P1.SEED, "top_subset": TOP_SUBSET}}
    inputs = {"primary weights": prim.file("weights.parquet"), "codeonly weights": code.file("weights.parquet"),
              "V-reader-LM weights": lmr.file("weights.parquet"), "code features": gen / "code_features.parquet",
              "TA-lite answers": root / TAL / "answers.parquet", "Mono-CoT answers": root / MCOT / "answers.parquet",
              "VIXCLS": root / "data/raw/macro/fred/VIXCLS.json", "registered pd1": root / "results/analysis/pd1.json"}
    p = X.write(NAME, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv,
                why="rows, columns, stress table and subset books specified after the unlock (beyond the registered "
                    "PD1 table)")
    for k in order:
        r = rows[k]
        print(f"{r['label']:<34} SR {r['sharpe']['sharpe']:.2f} CAGR {r['cagr']:.3f} MDD {r['max_drawdown']:.3f}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
