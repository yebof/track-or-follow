#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): macros and appendix tables from results/exploratory/*.json.

    .venv/bin/python experiments/exploratory/make_exploratory.py [--check]

Writes paper/generated/exploratory_numbers.tex (PD3x, PD1x and, since 2026-09-29, E1-E4 from e{1,2,3,4}_*.py; every
macro is prefixed \\ex and its comment carries the label
"exploratory (not pre-registered)"), and the repository tables results/exploratory/tables/
tableX1_portfolio_exploratory.tex and tableX2_subset_exploratory.tex (each note carries the label; the extra books and
the stress table go to the repository, not the paper: design_v2_delta PE-6, PE-16). It reads only files whose `meta`
says exploratory, so it can never present a registered number as its own or an exploratory one as registered; the
registered macros stay in paper/generated/numbers.tex (experiments/report/make_numbers.py). Numbers are never typed
by hand (CLAUDE.md rule 2).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
import _latex as L  # noqa: E402  (registered formatting helpers, read-only)
from make_numbers import fmt_num, half_up  # noqa: E402  (read-only)

LABEL = "exploratory (not pre-registered)"
XDIR = ROOT / "results" / "exploratory"
OUT = ROOT / "paper" / "generated" / "exploratory_numbers.tex"
TABLES = ROOT / "results" / "exploratory" / "tables"  # repository tables (PE-6, PE-16); the paper does not input them


def load(name: str) -> dict | None:
    p = XDIR / f"{name}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    meta = d.get("meta", {})
    if not (meta.get("exploratory") is True and meta.get("label") == LABEL and (meta.get("lock") or {}).get("open")):
        raise SystemExit(f"{p}: not an exploratory output written with the lock open")
    return d


def walk(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def num(x, dec=3, scale=1.0):
    return None if not L.finite(x) else half_up(x * scale, dec)


def ci(v, dec=3, scale=1.0):
    if not (isinstance(v, list) and len(v) == 2 and all(L.finite(x) for x in v)):
        return None
    return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]"


def macros(pd3: dict | None, px: dict | None) -> list[tuple[str, str | None, str]]:
    m: list[tuple[str, str | None, str]] = []
    c = walk(pd3, "comparisons") or {}
    for key, tag in (("mepi_off", "Epi"), ("mwork_off", "Work"), ("all_off", "All")):
        v = c.get(key) or {}
        src = f"results/exploratory/pd3_stores_off.json comparisons.{key}"
        m += [(f"exPdThree{tag}DeltaIC", num(walk(v, "delta_ic", "mean"), 4), f"PD3x {key}: dIC full - off, S30 | {src}"),
              (f"exPdThree{tag}DeltaICCI", ci(walk(v, "delta_ic", "ci"), 4), f"PD3x {key}: NW(4) 95% CI of dIC | {src}"),
              (f"exPdThree{tag}DeltaLL", num(walk(v, "delta_log_loss", "mean"), 5),
               f"PD3x {key}: weekly log loss(off) - loss(full) | {src}"),
              (f"exPdThree{tag}DeltaLLCI", ci(walk(v, "delta_log_loss", "ci"), 5),
               f"PD3x {key}: NW(4) 95% CI of the log-loss difference | {src}"),
              (f"exPdThree{tag}DeltaIR", num(walk(v, "delta_ir_sized", "diff"), 2),
               f"PD3x {key}: IR(full) - IR(off) of D-sized vs its EW-U100 | {src}"),
              (f"exPdThree{tag}DeltaIRCI", ci(walk(v, "delta_ir_sized", "ci"), 2),
               f"PD3x {key}: block-bootstrap 95% CI of the IR difference | {src}"),
              (f"exPdThree{tag}Weeks", fmt_num(int(v["n_weeks_ic"])) if L.finite(v.get("n_weeks_ic")) else None,
               f"PD3x {key}: learned weeks compared | {src}")]
    s = walk(px, "subset_s30_last60") or {}
    src = "results/exploratory/pd1_extended.json subset_s30_last60"
    m += [("exSubsetWeeks", fmt_num(int(s["n_weeks"])) if L.finite(s.get("n_weeks")) else None,
           f"TA-lite / Mono-CoT subset: decision weeks | {src}"),
          ("exTalEmptyWeeks", fmt_num(int(walk(s, "empty_weeks", "tal_native"))) if L.finite(
              walk(s, "empty_weeks", "tal_native")) else None, f"weeks with an empty TA-lite native book | {src}")]
    for key, tag in (("tal", "Tal"), ("mcot", "Mcot"), ("typed", "Typed")):
        m += [(f"exSubsetIC{tag}", num(walk(s, "ic", key, "mean"), 3), f"subset weekly rank-IC, {key} | {src}"),
              (f"exSubsetIC{tag}CI", ci(walk(s, "ic", key, "ci"), 3), f"NW(4) 95% CI of that IC | {src}")]
    st = walk(px, "stress", "mechanical") or {}
    m += [("exStressWeeks", fmt_num(int(st["n_weeks"])) if L.finite(st.get("n_weeks")) else None,
           "mechanical stress weeks (SPY bottom decile or VIXCLS > 30) | results/exploratory/pd1_extended.json")]
    rows = walk(px, "rows") or {}
    for key, tag in (("sized", "Sized"), ("sel_ew", "SelEW"), ("ew_u100", "EW")):
        r = rows.get(key) or {}
        m += [(f"exCalmar{tag}", num(r.get("calmar"), 2), f"Calmar (CAGR / |MDD|), {key} | pd1_extended.json rows"),
              (f"exCostDrag{tag}", num(r.get("cost_drag_ann"), 1, 100.0),
               f"cost drag (%/yr, 0 bps minus 15 bps), {key} | pd1_extended.json rows")]
        if key != "ew_u100":
            m += [(f"exIR{tag}", num(walk(r, "ir_vs_ew", "sharpe"), 2), f"IR vs EW-U100, {key} | pd1_extended.json"),
                  (f"exIR{tag}CI", ci(walk(r, "ir_vs_ew", "ci"), 2), f"95% CI of that IR | pd1_extended.json")]
    return m


def macros_e(e1: dict | None, e2: dict | None, e3: dict | None, e4: dict | None) -> list[tuple[str, str | None, str]]:
    """E1-E4 (experiments/exploratory/e{1,2,3,4}_*.py; exploratory analyses written on 2026-09-29)."""
    m: list[tuple[str, str | None, str]] = []
    s1 = "results/exploratory/e1_asr_mechanism.json"
    for r, tag in (("typed", "Typed"), ("H", "H")):
        m += [(f"exEoneInfl{tag}", num(walk(e1, "clean_influence", r, "sd_LWbar_E_over_kappa", "mean"), 2),
               f"E1 clean text influence of {r}: mean weekly sd over E_t of the capped Lbar^W / kappa | {s1}"),
              (f"exEonePivotal{tag}Pct", num(walk(e1, "clean_influence", r, "pivotal_share_E", "mean"), 1, 100.0),
               f"E1 % of E_t names whose D-sel-EW membership flips with their own text logit set to 0, {r} | {s1}"),
              (f"exEoneASRperInfl{tag}", num(walk(e1, "asr_per_unit_influence", r, "estimate"), 3),
               f"E1 ASR / mean influence (sd Lbar^W / kappa at the unit's week), {r} | {s1}")]
    m += [("exEoneSlopeHOutlook", num(walk(e1, "slopes", "H", "gamma", "h_outlook", "mean"), 4),
           f"E1 mean learned slope gamma of h_outlook over learned weeks (V-reader-H) | {s1}"),
          ("exEoneSlopeHOutlookPosPct", num(walk(e1, "slopes", "H", "gamma", "h_outlook", "share_weeks_positive"), 0,
                                            100.0), f"E1 % of learned weeks with gamma(h_outlook) > 0 | {s1}"),
          ("exEoneHOutlookUpPct", num(walk(e1, "attack_shift", "H", "own_feature", "share_own_raw_up"), 0, 100.0),
           f"E1 % of attacked units whose raw h_outlook rose | {s1}"),
          ("exEoneHDisp", num(walk(e1, "attack_shift", "H", "all", "rank_displacement_mean"), 2),
           f"E1 mean combined-rank displacement of the attacked name, H (positive = moved up) | {s1}")]
    for r, tag in (("H", "H"), ("typed", "Typed")):
        for key, t2 in (("matched_to_typed", "MatchedTyped"), ("matched_to_H", "MatchedH"),
                        ("flip_to_typed", "FlipTyped")):
            v = walk(e1, "asr", r, key)
            if v is None:
                continue
            m += [(f"exEoneASR{tag}{t2}", num(v.get("asr"), 1, 100.0), f"E1 ASR (%) of {r}, replay {key} | {s1}"),
                  (f"exEoneASR{tag}{t2}CI", ci(v.get("ci"), 1, 100.0), f"E1 cluster-bootstrap 95% CI (%) | {s1}")]
    tf = walk(e1, "typed_features", "features") or {}
    for f, tag in (("no_news", "NoNews"), ("f_und", "FUnd"), ("f_net", "FNet")):
        v = tf.get(f) or {}
        m += [(f"exEoneKO{tag}ASR", num(walk(v, "knockout_asr", "asr"), 1, 100.0),
               f"E1 typed ASR (%) with {f}'s attacked values reverted to clean (all names of the week) | {s1}"),
              (f"exEoneKO{tag}Lost", fmt_num(int(v["successes_lost_by_knockout"])) if L.finite(
                  v.get("successes_lost_by_knockout")) else None,
               f"E1 typed successes lost when {f} is reverted | {s1}"),
              (f"exEoneLargest{tag}Pct", num(v.get("share_successes_largest_positive_contributor"), 0, 100.0),
               f"E1 % of typed successes with {f} the largest positive contributor to dL^W | {s1}")]
    m += [("exEoneTypedSuccesses", fmt_num(int(walk(e1, "typed_features", "n_successes")))
           if L.finite(walk(e1, "typed_features", "n_successes")) else None,
           f"E1 typed successful promotion attacks (repeats h1.json's 117) | {s1}")]
    s2 = "results/exploratory/e2_priced_horizon.json"
    for key, tag, path in (("ar_2_5", "ARtwofive", ("item_level", "windows_event_time", "ar_2_5")),
                           ("ar_6_10", "ARsixten", ("item_level", "windows_event_time", "ar_6_10")),
                           ("prefill", "PreFill", ("item_level", "pre_fill_ar_d0_to_dt")),
                           ("traded", "Traded", ("item_level", "traded_holding_week_r_minus_m")),
                           ("fnet_news", "FnetNewsWeek", ("stock_week_level", "f_net", "news_week_ar_m4_0")),
                           ("fnet_traded", "FnetTraded", ("stock_week_level", "f_net",
                                                          "traded_holding_week_r_minus_m"))):
        v = walk(e2, *path) or {}
        m += [(f"exEtwo{tag}", num(v.get("mean"), 3), f"E2 mean weekly Spearman, {key} | {s2}"),
              (f"exEtwo{tag}CI", ci(v.get("ci"), 3), f"E2 two-sided NW(4) 95% CI, {key} | {s2}")]
    m += [("exEtwoClosedBeforeFillPct", num(walk(e2, "timing", "share_window_0_1_closed_before_fill"), 0, 100.0),
           f"E2 % of H2 items whose AR[0,+1] window closes before the fill (open e_t) | {s2}")]
    s3 = "results/exploratory/e3_lookahead_by_month.json"
    fl = walk(e3, "d1", "registered_probe_period", "windows_differing_from_0_5") or []
    m += [("exEthreeFlaggedFrom", fl[0] if fl else None, f"E3 first D1 window centre differing from 0.5 | {s3}"),
          ("exEthreeFlaggedTo", fl[-1] if fl else None, f"E3 last D1 window centre differing from 0.5 | {s3}")]
    for per, tag in (("pre_release_complement", "Pre"), ("W_release", "Rel"), ("d1_flagged_months", "Flag"),
                     ("not_d1_flagged", "Rest")):
        for key, t2, dec in (("h2_weekly_spearman", "Htwo", 3), ("h3_delta_ic", "Hthree", 3),
                             ("h3_ic_primary", "IC", 3)):
            v = walk(e3, "h2_h3", "by_period", per, key) or {}
            m += [(f"exEthree{t2}{tag}", num(v.get("mean"), dec), f"E3 {key}, {per} | {s3}"),
                  (f"exEthree{t2}{tag}CI", ci(v.get("ci"), dec), f"E3 two-sided NW(4) 95% CI, {key}, {per} | {s3}")]
    s4 = "results/exploratory/e4_trader_attribution.json"
    A = walk(e4, "decomposition", "A_selection_overlay_text") or {}
    m += [("exEfourActive", num(walk(e4, "active_total", "net_bps_per_week", "mean"), 1),
           f"E4 D-sized minus EW-U100, net, bps/week | {s4}"),
          ("exEfourActiveCI", ci(walk(e4, "active_total", "net_bps_per_week", "ci"), 1),
           f"E4 NW(4) 95% CI | {s4}")]
    for layer, tag in (("trusted_selection", "Selection"), ("overlay", "Overlay"), ("text", "Text")):
        m += [(f"exEfour{tag}", num(walk(A, layer, "net_bps_per_week", "mean"), 1),
               f"E4 order A layer {layer}, net bps/week | {s4}"),
              (f"exEfour{tag}CI", ci(walk(A, layer, "net_bps_per_week", "ci"), 1), f"E4 NW(4) 95% CI | {s4}"),
              (f"exEfour{tag}Gross", num(walk(A, layer, "gross_bps_per_week", "mean"), 1),
               f"E4 order A layer {layer}, gross (0 bps) bps/week | {s4}")]
    m += [("exEfourOverlayCash", num(walk(A, "overlay_split", "uninvested_share_part_bps_per_week", "mean"), 1),
           f"E4 overlay part from the uninvested share, bps/week | {s4}"),
          ("exEfourOverlayReweight", num(walk(A, "overlay_split", "reweighting_part_bps_per_week", "mean"), 1),
           f"E4 overlay part from re-weighting the invested share, bps/week | {s4}"),
          ("exEfourCashPct", num(walk(A, "overlay_split", "uninvested_share_mean"), 0, 100.0),
           f"E4 mean uninvested share of the trusted-only sized book, % | {s4}"),
          ("exEfourTurnoverSized", num(walk(e4, "books", "s_sized", "turnover_one_way_per_rebalance"), 0, 100.0),
           f"E4 one-way turnover per rebalance of D-sized, % | {s4}"),
          ("exEfourCostDragSized", num(walk(e4, "books", "s_sized", "cost_drag_annualized"), 1, 100.0),
           f"E4 annualized cost drag of D-sized, %/yr | {s4}"),
          ("exEfourEntrantsPerWeek", num(walk(e4, "text_names", "entrants_per_week_mean"), 1),
           f"E4 mean names per week the text moved into D-sel-EW (vs the trusted top-K) | {s4}")]
    return m


def tex_macros(m) -> str:
    lines = ["% paper/generated/exploratory_numbers.tex -- GENERATED by experiments/exploratory/make_exploratory.py.",
             f"% Every macro below is {LABEL}: never a confirmatory or registered number. Do not edit by hand.",
             "\\providecommand{\\NumPlaceholder}[1]{\\textbf{[#1]}}"]
    for name, val, comment in m:
        if re.search(r"[^A-Za-z]", name) or not name.startswith("ex"):
            raise SystemExit(f"bad exploratory macro name {name}")
        lines += [f"% {LABEL} | {comment}",
                  f"\\newcommand{{\\{name}}}{{{val if val is not None else '\\NumPlaceholder{' + name + '}'}}}"]
    return "\n".join(lines) + "\n"


def row_cells(r: dict, stress: dict | None, key: str, wide: bool = True) -> list[str]:
    sh = r.get("sharpe") or {}
    ir = r.get("ir_vs_ew") or {}
    ff = r.get("ff5_mom") or {}
    cells = [L.pct(r.get("cagr"), 1), L.est_ci(sh.get("sharpe"), sh.get("ci"), 2), L.pct(r.get("max_drawdown"), 1),
             L.num(r.get("calmar"), 2), L.pct(r.get("turnover_one_way"), 0), L.num(r.get("cost_drag_ann"), 1, 100.0),
             L.est_ci(ir.get("sharpe"), ir.get("ci"), 2) if ir else L.NA]
    if wide:
        cells.append(L.est_ci(ff.get("alpha_annualized"), ff.get("ci"), 1, 100.0) if ff else L.NOT_RUN)
        cells.append(L.num(ff.get("beta_mkt"), 2) if ff else L.NOT_RUN)
        for ep in ("tariff_2025_04", "tariff_2025_10"):
            v = walk(stress, "named", ep, "books", key, "cum_return")
            cells.append(L.num(v, 1, 100.0))
        cells.append(L.num(walk(stress, "mechanical", "books", key, "mean_weekly_return"), 2, 100.0))
    return cells


def table_x1(px: dict) -> str:
    body = [L.row(px["rows"][k]["label"], *row_cells(px["rows"][k], px.get("stress"), k)) for k in px["order"]]
    rnd = walk(px, "registered_reference", "random20") or {}
    if rnd.get("sharpe_p50") is not None:
        body.append(L.row(f"Random-20 (median of {rnd.get('n')})", L.NA,
                          f"{L.num(rnd.get('sharpe_p50'), 2)} {{\\tiny [{L.num(rnd.get('sharpe_p05'), 2)}, "
                          f"{L.num(rnd.get('sharpe_p95'), 2)}]}}", *[L.NA] * 10))
    header = ("& CAGR & Sharpe [CI] & MDD & Calmar & TO & Drag & IR vs EW [CI] & FF5+Mom $\\alpha$ [CI] & "
              "$\\beta_{\\mathrm{mkt}}$ & Apr 2--9 '25 & Oct 10 '25 & Stress wk \\\\")
    st = walk(px, "stress", "mechanical") or {}
    ff_last = None
    for r in px["rows"].values():
        ff_last = walk(r, "ff5_mom", "coverage", "factor_last_date") or ff_last
    note = (f"\\textbf{{{LABEL.capitalize()}.}} U100, {px.get('n_decisions')} weekly decisions, net of 15~bps per side "
            f"(registered PD1 conventions). CAGR, MDD, TO (one-way turnover per rebalance), Drag (annualized return at "
            f"0~bps minus 15~bps), episode returns and the mean return in the {st.get('n_weeks')} mechanical stress "
            f"weeks (SPY bottom decile or VIX above 30) in \\%; FF5+Mom $\\alpha$ in \\%/yr on the dates the French "
            f"data cover (to {ff_last}). Random-20: 5--95\\% range of 1,000 draws (registered PD1 output). Estimates, "
            f"not tests.")
    return L.tabular("l" + "r" * 12, header, body, note, size="\\scriptsize")


def table_x2(px: dict) -> str:
    s = px["subset_s30_last60"]
    body = [L.row(s["rows"][k]["label"], *row_cells(s["rows"][k], None, k, wide=False)) for k in s["order"]]
    ic = s.get("ic") or {}
    note = (f"\\textbf{{{LABEL.capitalize()}.}} S30 $\\times$ the last {s.get('n_weeks')} W$_{{\\mathrm{{clean}}}}$ "
            f"decisions ({s['dates'][0]} to {s['dates'][1]}), the stock-weeks TA-lite and Mono-CoT ran on; IR vs "
            f"EW-S30. TA-lite's native book is empty (cash) in {walk(s, 'empty_weeks', 'tal_native')} weeks. Weekly "
            f"rank-IC with $r-m$ on the same stock-weeks: TA-lite {L.est_ci(walk(ic, 'tal', 'mean'), walk(ic, 'tal', 'ci'), 3)}, "
            f"Mono-CoT {L.est_ci(walk(ic, 'mcot', 'mean'), walk(ic, 'mcot', 'ci'), 3)}, typed $L$ "
            f"{L.est_ci(walk(ic, 'typed', 'mean'), walk(ic, 'typed', 'ci'), 3)}.")
    header = "& CAGR & Sharpe [CI] & MDD & Calmar & TO & Drag & IR vs EW [CI] \\\\"
    return L.tabular("l" + "r" * 7, header, body, note, size="\\scriptsize")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    pd3, px = load("pd3_stores_off"), load("pd1_extended")
    es = [load(n) for n in ("e1_asr_mechanism", "e2_priced_horizon", "e3_lookahead_by_month", "e4_trader_attribution")]
    outs = {OUT: tex_macros(macros(pd3, px) + macros_e(*es))}
    if px:
        outs[TABLES / "tableX1_portfolio_exploratory.tex"] = table_x1(px)
        outs[TABLES / "tableX2_subset_exploratory.tex"] = table_x2(px)
    stale = [p for p, t in outs.items() if not p.is_file() or p.read_text() != t]
    if a.check:
        print("stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale) if stale else "exploratory: up to date")
        return 1 if stale else 0
    for p, t in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)} ({LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
