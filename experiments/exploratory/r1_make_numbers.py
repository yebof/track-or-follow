#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the \\ex macros added after the round-1 paper reviews (E6 publication timing and
H2 qualifiers, E7 reference readers, and the multiplicity of the registered D1 windows)
-> paper/generated/exploratory_numbers_r1.tex.

    .venv/bin/python experiments/exploratory/r1_make_numbers.py [--check]

`main.tex` inputs this file after the other exploratory macro files; it never redefines a name they define. Every macro
name starts with \\ex and every comment carries the label "exploratory (not pre-registered)"; a sentence that uses one
says "exploratory" in the same sentence (outline_v2 §5.5). Inputs come only through the exploratory loader
(experiments/report/_validity.py: meta.exploratory true, the label, the lock open), except the D1 multiplicity note,
which applies Holm's step-down to the registered per-window permutation p-values of results/analysis/d1.json (a new
computation on registered outputs, hence exploratory). A missing input leaves a bold [name] placeholder and exit 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
import _validity as V  # noqa: E402  (exploratory loader)
from make_numbers import fmt_num, half_up  # noqa: E402  (read-only)

OUT = ROOT / "paper" / "generated" / "exploratory_numbers_r1.tex"
LABEL = V.LABEL_X
S6 = "results/exploratory/e6_publication_timing.json"
S7 = "results/exploratory/e7_reference_readers.json"
TYPES = (("earnings_results", "Earnings"), ("analyst_action", "Analyst"), ("guidance_change", "Guidance"),
         ("product_or_contract", "Product"), ("management_or_labor", "Management"),
         ("industry_or_macro_readthrough", "Readthrough"))


def fin(x) -> bool:
    import math

    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def n(x, dec=3, scale=1.0, sign=False):
    if not fin(x):
        return None
    s = half_up(x * scale, dec)
    return ("+" + s) if sign and x > 0 else s


def c(v, dec=3, scale=1.0):
    if not (isinstance(v, list) and len(v) == 2 and all(fin(x) for x in v)):
        return None
    return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]"


def est(m, name, d, desc, src, key="mean", dec=3, scale=1.0):
    m.append((name, n(V.get(d, key), dec, scale), desc, src))
    m.append((name + "CI", c(V.get(d, "ci"), dec, scale), f"95% interval of {name}", src))


def macros_e6(e6: dict | None) -> list:
    m: list = []
    g = lambda *p: V.get(e6, *p)  # noqa: E731
    pa, tm, ct, pk, rs = (g(k) or {} for k in ("publication_aligned", "timing", "controls", "peakedness",
                                                "price_report_residual_approximation"))
    est(m, "exEsixPubAll", pa.get("all"), "E6 weekly Spearman of E[effect] with the publication-aligned AR (first trade "
        "after publication to the close of d0+1)", f"{S6} publication_aligned.all")
    est(m, "exEsixPubNoPrice", pa.get("no_price"), "the same without price-report items", f"{S6} publication_aligned.no_price")
    est(m, "exEsixPubRelease", pa.get("W_release"), "the same on W_release", f"{S6} publication_aligned.W_release")
    est(m, "exEsixPubRegular", pa.get("regular_entry_all"), "the same, entry at the first regular-session bar",
        f"{S6} publication_aligned.regular_entry_all")
    lag = pa.get("entry_lag_minutes") or {}
    m.append(("exEsixLagMedian", n(lag.get("median"), 1), "median minutes from publication to the entry bar",
              f"{S6} publication_aligned.entry_lag_minutes.median"))
    m.append(("exEsixLagFivePct", n(lag.get("share_le_5"), 0, 100), "% of items entered within 5 minutes",
              f"{S6} publication_aligned.entry_lag_minutes.share_le_5"))
    for key, w in (("pre_open", "PreOpen"), ("intraday", "Intraday"), ("after_close_or_nonsession", "After")):
        t = tm.get(key) or {}
        m.append((f"exEsixShare{w}", n(t.get("share"), 0, 100), f"% of frame items in timing group {key}",
                  f"{S6} timing.{key}.share"))
        est(m, f"exEsix{w}", t.get("ar_0_1"), f"weekly Spearman with AR[0,+1], timing group {key}",
            f"{S6} timing.{key}.ar_0_1")
        est(m, f"exEsix{w}ARzero", t.get("ar_0"), f"the same with AR[0], {key}", f"{S6} timing.{key}.ar_0")
        est(m, f"exEsix{w}Pub", t.get("ar_pub"), f"the same with the publication-aligned AR, {key}",
            f"{S6} timing.{key}.ar_pub")
    for key, w in TYPES:
        t = (pa.get("by_type") or {}).get(key) or {}
        est(m, f"exEsixPub{w}", t.get("ar_pub"), f"publication-aligned statistic, top event type {key}",
            f"{S6} publication_aligned.by_type.{key}.ar_pub")
    est(m, "exEsixRelNoPrice", ct.get("W_release_no_price"), "registered statistic on W_release without price-report "
        "items", f"{S6} controls.W_release_no_price")
    est(m, "exEsixRelPreOpenNoPrice", ct.get("W_release_pre_open_no_price"), "the same, pre-open items only",
        f"{S6} controls.W_release_pre_open_no_price")
    est(m, "exEsixPlaceboDiff", ct.get("h2_minus_placebo_weekly"), "paired weekly H2 minus placebo",
        f"{S6} controls.h2_minus_placebo_weekly")
    est(m, "exEsixPartial", ct.get("partial_given_placebo"), "weekly partial Spearman given AR[-5,-1]",
        f"{S6} controls.partial_given_placebo")
    m += [("exEsixNearInteger", n(pk.get("share_within_0.05_of_integer"), 1, 100), "% of E[effect] within 0.05 of an "
           "integer level", f"{S6} peakedness"),
          ("exEsixHelpful", n(pk.get("share_helpful_2.5_plus"), 1, 100), "% of items with E[effect] >= 2.5",
           f"{S6} peakedness"),
          ("exEsixNeutral", n(pk.get("share_neutral_1.5_to_2.5"), 1, 100), "% with 1.5 <= E[effect] < 2.5",
           f"{S6} peakedness"),
          ("exEsixHarmful", n(pk.get("share_harmful_below_1.5"), 1, 100), "% with E[effect] < 1.5", f"{S6} peakedness"),
          ("exEsixEntropy", n(pk.get("mean_normalized_entropy"), 3), "mean normalized entropy of the effect "
           "distributions", f"{S6} peakedness")]
    est(m, "exEsixRounded", pk.get("rounded_level_vs_ar"), "registered statistic with the rounded level",
        f"{S6} peakedness.rounded_level_vs_ar")
    m += [("exEsixResidual", n(rs.get("approx_residual"), 2), "approximate statistic on items that report no move, "
           "linear mixture with the audit's miss rate", f"{S6} price_report_residual_approximation"),
          ("exEsixResidualRange", c(rs.get("approx_residual_at_miss_ci"), 2), "the same at the ends of the miss rate's "
           "95% interval", f"{S6} price_report_residual_approximation"),
          ("exEsixMissRate", n(rs.get("miss_rate"), 0, 100), "audit: % of unflagged items that report the move",
           f"{S6} price_report_residual_approximation")]
    return m


def macros_e7(e7: dict | None) -> list:
    m: list = []
    f = V.get(e7, "finbert") or {}
    v = V.get(e7, "verb_minus_emu_same_units") or {}
    est(m, "exEsevenFinbert", f.get("all"), "FinBERT tone vs AR[0,+1], weekly Spearman", f"{S7} finbert.all")
    est(m, "exEsevenFinbertNoPrice", f.get("no_price"), "the same without price-report items", f"{S7} finbert.no_price")
    est(m, "exEsevenFinbertAUC", f.get("item_auc_all"), "FinBERT tone vs the sign of AR[0,+1], item AUC",
        f"{S7} finbert.item_auc_all", key="auc")
    m.append(("exEsevenFinbertAgree", n(f.get("agreement_with_emu_spearman"), 2), "Spearman of FinBERT tone with emu's "
              "E[effect]", f"{S7} finbert.agreement_with_emu_spearman"))
    m.append(("exEsevenVerbDiff", n(v.get("difference"), 3, sign=True), "pooled Spearman, VERB minus emu, same units",
              f"{S7} verb_minus_emu_same_units.difference"))
    m.append(("exEsevenVerbDiffCI", c(v.get("ci")), "its week-block 95% interval", f"{S7} verb_minus_emu_same_units.ci"))
    return m


def macros_d1() -> list:
    d1 = V.registered("d1")
    w = (d1 or {}).get("monthly_windows") or []
    ps = sorted(x["p_perm"] for x in w if fin(x.get("p_perm")))
    if not ps:
        return [("exDOneMinP", None, "", ""), ("exDOneHolmMinP", None, "", "")]
    k = len(ps)
    holm = min(1.0, ps[0] * k)
    return [("exDOneMinP", n(ps[0], 3), "smallest per-window permutation p of the registered D1 windows",
             "results/analysis/d1.json monthly_windows[*].p_perm"),
            ("exDOneHolmMinP", n(holm, 2), "its Holm-adjusted value over the registered windows (step 1)",
             "results/analysis/d1.json monthly_windows[*].p_perm")]


def render() -> tuple[str, list[str]]:
    e6, e7 = V.exploratory("e6_publication_timing"), V.exploratory("e7_reference_readers")
    rows = macros_e6(e6) + macros_e7(e7) + macros_d1()
    lines = ["% paper/generated/exploratory_numbers_r1.tex -- GENERATED by experiments/exploratory/r1_make_numbers.py.",
             f"% Every macro below is {LABEL}: never a confirmatory or registered number. Do not edit by hand.",
             "\\providecommand{\\NumPlaceholder}[1]{\\textbf{[#1]}}"]
    missing = []
    for name, v, desc, src in rows:
        if v is None:
            missing.append(name)
            v = f"\\NumPlaceholder{{{name}}}"
        lines += [f"% {LABEL} | {desc} | {src}", f"\\newcommand{{\\{name}}}{{{v}}}"]
    return "\n".join(lines) + "\n", missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text, missing = render()
    if a.check:
        stale = not OUT.is_file() or OUT.read_text() != text
        print("exploratory_numbers_r1: " + ("stale" if stale else "up to date") +
              (f"; missing: {', '.join(missing)}" if missing else ""))
        return 1 if stale or missing else 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}" + (f"; MISSING (placeholders): {', '.join(missing)}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
