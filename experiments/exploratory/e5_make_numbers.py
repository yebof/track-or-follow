#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the \\ex macros of the content-analysis package (E5 construct checks, Box 1)
and the E3 additions of outline_v2 §5.3 -> paper/generated/exploratory_numbers_extra.tex.

    .venv/bin/python experiments/exploratory/e5_make_numbers.py [--check]

`main.tex` inputs this file after generated/exploratory_numbers.tex (experiments/exploratory/make_exploratory.py,
which this script does not edit). Every macro name starts with \\ex and every comment carries the label
"exploratory (not pre-registered)"; a sentence that uses one says "exploratory (not pre-registered)" in the same
sentence (outline_v2 §5.5). Inputs are read only through the exploratory loader (meta.exploratory true, the label, the
lock open), so a registered number can never be printed as exploratory or the reverse. Placeholders:
\\exFidReaskFloor and \\exFidJointAgree belong to the FID summary (D-11), which is not part of this package; they are
\\providecommand placeholders here (a later definition in exploratory_numbers.tex wins) and print as bold [name].
The optional either-classifier sensitivity (spec v1.1) prints only once every frame text is coded.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
import _validity as V  # noqa: E402  (exploratory loader)
from make_numbers import fmt_num, half_up  # noqa: E402  (read-only)

OUT = ROOT / "paper" / "generated" / "exploratory_numbers_extra.tex"
LABEL = V.LABEL_X
S5 = "results/exploratory/e5_construct_checks.json"
SB = "results/exploratory/e5_box1_examples.json"
S3 = "results/exploratory/e3_lookahead_by_month.json"
CONCEPTS = (("earnings_results", "Earnings"), ("analyst_action", "Analyst"), ("guidance_change", "Guidance"),
            ("management_or_labor", "Management"), ("industry_or_macro_readthrough", "Readthrough"),
            ("other", "Other"), ("product_or_contract", "Product"), ("legal_or_regulatory", "Legal"),
            ("deal_or_investment", "Deal"), ("operations_disruption", "Operations"),
            ("financing_or_credit", "Financing"), ("capital_return", "Capital"))


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


def integer(x):
    return fmt_num(int(x)) if fin(x) else None


def est(m, name, v, desc, src, dec=3, scale=1.0, ci=True):
    """An estimate and (optionally) its CI from a {'mean'|..., 'ci'} dict."""
    key = next((k for k in ("mean", "auc", "hit_rate") if isinstance(v, dict) and k in v), None)
    m.append((name, n(V.get(v, key), dec, scale) if key else None, desc, src))
    if ci:
        m.append((name + "CI", c(V.get(v, "ci"), dec, scale), f"95% interval of {name}", src))


def macros_e5(e5: dict | None, box: dict | None) -> list[tuple[str, str | None, str, str]]:
    m: list[tuple[str, str | None, str, str]] = []
    g = lambda *p: V.get(e5, *p)  # noqa: E731
    pr = g("price_report") or {}
    m += [("exEfivePriceShare", n(pr.get("share_units"), 1, 100), "E5 % of H2 units whose text reports the price move "
           "(regex of record)", f"{S5} price_report.share_units"),
          ("exEfivePriceN", integer(pr.get("n_units")), "E5 H2 units whose text reports the price move", S5),
          ("exEfivePriceShareHead", n(pr.get("share_units_headline"), 1, 100), "E5 % of units whose headline reports "
           "the move", f"{S5} price_report.share_units_headline"),
          ("exEfivePriceSharePreOpen", n(pr.get("share_pre_open_units"), 1, 100),
           "E5 % of pre-open units whose text reports the move", f"{S5} price_report.share_pre_open_units")]
    coder = g("audit", g("audit", "paper_coder") or "model_coder") or {}
    ok = coder.get("status") == "ok"
    src = f"{S5} audit.{g('audit', 'paper_coder')}"
    m += [("exEfiveAuditPrecision", n(coder.get("precision"), 0, 100) if ok else None,
           "E5 audit: % of audited regex matches the coder codes as reporting the move (precision)", src),
          ("exEfiveAuditPrecisionCI", c(coder.get("precision_ci"), 0, 100) if ok else None, "Wilson 95% interval", src),
          ("exEfiveAuditRecall", n(coder.get("recall"), 0, 100) if ok else None,
           "E5 audit: estimated recall of the regex (%), stratum-weighted", src),
          ("exEfiveAuditRecallCI", c(coder.get("recall_ci"), 0, 100) if ok else None,
           "stratified bootstrap 95% interval", src),
          ("exEfiveAuditMiss", n(coder.get("miss_rate"), 0, 100) if ok else None,
           "E5 audit: % of audited non-matches the coder codes as reporting the move", src),
          ("exEfiveAuditPrevalence", n(coder.get("coder_prevalence"), 1, 100) if ok else None,
           "E5 audit: coder-implied % of distinct frame texts that report the move", src),
          ("exEfiveAuditN", integer(coder.get("n_match_coded")) if ok else None,
           "E5 audit: items coded per stratum (matches)", src),
          ("exEfiveAuditCoder", ("a human coder" if "human" in str(coder.get("coder")) else "the open typed model")
           if ok else None, "E5 audit: who coded (text, not a number)", src)]
    w = g("weekly") or {}
    for key, tag in (("no_price", "NoPrice"), ("price_only", "PriceOnly"), ("pre_open_no_price", "PreOpenNoPrice"),
                     ("no_price_headline", "NoPriceHead"), ("placebo_no_price", "PlaceboNoPrice")):
        est(m, f"exEfive{tag}", w.get(key), f"E5 weekly Spearman, {key}", f"{S5} weekly.{key}")
    m += [("exEfiveNoPriceN", integer(V.get(w, "no_price", "n_units")), "E5 units without a price report", S5)]
    d = g("dictionary_reference") or {}
    est(m, "exEfiveLM", d.get("all"), "E5 Loughran-McDonald tone, weekly Spearman, all units",
        f"{S5} dictionary_reference.all")
    est(m, "exEfiveLMNoPrice", d.get("no_price"), "E5 LM tone, units without a price report",
        f"{S5} dictionary_reference.no_price")
    a = g("item_auc") or {}
    est(m, "exEfiveAUCSign", a.get("E_effect_all"), "E5 item AUC of E[effect] vs 1[AR[0,+1] > 0]", f"{S5} item_auc")
    est(m, "exEfiveAUCSignNoPrice", a.get("E_effect_no_price"), "E5 the same without price reports", f"{S5} item_auc")
    est(m, "exEfiveLMAUCSign", a.get("lm_all"), "E5 item AUC of the LM tone vs the sign of AR[0,+1]", f"{S5} item_auc")
    h = g("hit_rate") or {}
    est(m, "exEfiveHitRate", h.get("all"), "E5 directional hit rate (%), E[effect] outside [1.5, 2.5]",
        f"{S5} hit_rate.all", dec=1, scale=100)
    est(m, "exEfiveHitRateNoPrice", h.get("no_price"), "E5 the same without price reports (%)",
        f"{S5} hit_rate.no_price", dec=1, scale=100)
    m += [("exEfiveHitRateN", integer(V.get(h, "all", "n")), "E5 units of the hit rate", S5)]
    fd = g("firm_day") or {}
    est(m, "exEfiveFirmDay", fd, "E5 firm-day collapse, weekly Spearman", f"{S5} firm_day")
    m += [("exEfiveFirmDays", integer(fd.get("firm_days")), "E5 firm-days", S5)]
    bc = g("by_concept") or {}
    for key, tag in CONCEPTS:
        v = bc.get(key) or {}
        est(m, f"exEfiveConcept{tag}", v.get("all"), f"E5 weekly Spearman within top event type {key}",
            f"{S5} by_concept.{key}.all")
        est(m, f"exEfiveConcept{tag}NoPrice", v.get("no_price"), f"E5 the same without price reports, {key}",
            f"{S5} by_concept.{key}.no_price")
        m += [(f"exEfiveConcept{tag}Share", n(v.get("share"), 1, 100), f"E5 % of H2 units with top type {key}",
               f"{S5} by_concept.{key}.share")]
    s = g("either_classifier_sensitivity") or {}
    if "no_price_either" in s:
        m += [("exEfiveCoderShare", n(s.get("share_units_coder"), 1, 100), "E5 v1.1 % of units the coder flags", S5),
              ("exEfiveEitherShare", n(s.get("share_units_either"), 1, 100), "E5 v1.1 % flagged by regex or coder",
               S5)]
        est(m, "exEfiveNoPriceEither", s.get("no_price_either"), "E5 v1.1 weekly Spearman, flagged by neither", S5)
        est(m, "exEfivePreOpenNoPriceEither", s.get("pre_open_no_price_either"),
            "E5 v1.1 pre-open and flagged by neither", S5)
    # Box 1 (illustration)
    for tag in ("A", "B"):
        it = V.get(box, "items", tag) or {}
        m += [(f"exBox{tag}ARzeroOne", n(it.get("ar_0_1"), 1, 100, sign=True), f"Box 1 item {tag}: AR[0,+1] (%)", SB),
              (f"exBox{tag}ARplacebo", n(it.get("ar_m5_m1"), 1, 100, sign=True), f"Box 1 item {tag}: AR[-5,-1] (%)",
               SB),
              (f"exBox{tag}Effect", n(it.get("E_effect"), 2), f"Box 1 item {tag}: E[effect]", SB)]
    return m


def macros_e3(e3: dict | None) -> list[tuple[str, str | None, str, str]]:
    """E3 additions (outline_v2 §5.3): the W_pre-slice D1 windows wholly before the documented cutoff."""
    wins = V.get(e3, "d1", "w_pre_slice", "pooled_windows") or []
    cut = next((x["date"] for x in (V.get(e3, "dates") or []) if "cutoff" in x.get("label", "")), None)
    alpha = V.get(e3, "d1", "settings", "alpha")
    if not wins or cut is None or not fin(alpha):
        return [(k, None, "E3 addition", S3) for k in ("exEthreePreCutoffWindows", "exEthreePreCutoffFlagged",
                                                        "exEthreePreCutoffAboveHalf", "exEthreePreCutoffMeanAUC")]
    pre = [w for w in wins if all(f"{mm}-01" < cut for mm in w["months"])]
    return [("exEthreePreCutoffWindows", integer(len(pre)), "E3 D1 3-month windows wholly before the documented "
             "cutoff (W_pre slice, development run)", f"{S3} d1.w_pre_slice.pooled_windows"),
            ("exEthreePreCutoffFlagged", integer(sum(w["p_perm"] < alpha for w in pre)),
             "E3 of those, windows whose AUC lies outside the permutation null band", S3),
            ("exEthreePreCutoffAboveHalf", integer(sum(w["auc"] > 0.5 for w in pre)), "E3 of those, AUC above 0.5", S3),
            ("exEthreePreCutoffMeanAUC", n(sum(w["auc"] for w in pre) / len(pre), 3), "E3 their mean AUC", S3)]


def render() -> tuple[str, list[str]]:
    e5, box, e3 = V.exploratory("e5_construct_checks"), V.exploratory("e5_box1_examples"), \
        V.exploratory("e3_lookahead_by_month")
    m = macros_e5(e5, box) + macros_e3(e3)
    lines = ["% paper/generated/exploratory_numbers_extra.tex -- GENERATED by experiments/exploratory/e5_make_numbers.py.",
             f"% Every macro below is {LABEL}: never a confirmatory or registered number. Do not edit by hand.",
             "\\providecommand{\\NumPlaceholder}[1]{\\textbf{[#1]}}"]
    missing = []
    for name, val, desc, src in m:
        if re.search(r"[^A-Za-z]", name) or not name.startswith("ex"):
            raise SystemExit(f"bad exploratory macro name {name}")
        if val is None:
            missing.append(name)
            val = f"\\NumPlaceholder{{{name}}}"
        lines += [f"% {LABEL} | {desc} | {src}", f"\\newcommand{{\\{name}}}{{{val}}}"]
    lines += [f"% {LABEL} | FID summary (D-11), not part of this package: placeholder unless defined earlier",
              "\\providecommand{\\exFidReaskFloor}{\\NumPlaceholder{exFidReaskFloor}}",
              "\\providecommand{\\exFidJointAgree}{\\NumPlaceholder{exFidJointAgree}}"]
    return "\n".join(lines) + "\n", missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text, missing = render()
    if a.check:
        stale = not OUT.is_file() or OUT.read_text() != text
        print("exploratory_numbers_extra: " + ("stale" if stale else "up to date") +
              (f"; placeholders: {', '.join(missing)}" if missing else ""))
        return 1 if stale else 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({LABEL}); placeholders: {', '.join(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
