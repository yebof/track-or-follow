#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the \\ex macros added after the round-2 paper reviews (E8 stress tests of the
timing and reference-reader findings, E8 frame sensitivities, one E2 row not printed before, and the count of
exploratory analyses) -> paper/generated/exploratory_numbers_r2.tex.

    .venv/bin/python experiments/exploratory/r2_make_numbers.py [--check]

`main.tex` inputs this file after the other exploratory macro files; it never redefines a name they define. Every macro
name starts with \\ex and every comment carries the label "exploratory (not pre-registered)"; a sentence that uses one
says "exploratory" in the same sentence (outline_v2 §5.5). Inputs come only through the exploratory loader
(experiments/report/_validity.py: meta.exploratory true, the label, the lock open). A missing input leaves a bold
[name] placeholder and exit 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402  (exploratory loader)
from make_numbers import fmt_num  # noqa: E402  (read-only)
from r1_make_numbers import c, est, n  # noqa: E402  (same formatting as the round-1 macros)

OUT = ROOT / "paper" / "generated" / "exploratory_numbers_r2.tex"
LABEL = V.LABEL_X
S8 = "results/exploratory/e8_timing_robustness.json"
S2 = "results/exploratory/e2_priced_horizon.json"


def count(x) -> str | None:
    return fmt_num(int(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def macros_e8(e8: dict | None) -> list:
    m: list = []
    g = lambda *p: V.get(e8, *p)  # noqa: E731
    pr, pdiff = g("readers", "per_reader") or {}, g("readers", "paired_differences") or {}
    fb, dc, em = pr.get("finbert") or {}, pr.get("dictionary") or {}, pr.get("emu") or {}
    est(m, "exEeightFinbertPub", fb.get("ar_pub"), "FinBERT tone vs the publication-aligned AR, weekly Spearman",
        f"{S8} readers.per_reader.finbert.ar_pub")
    est(m, "exEeightDictPub", dc.get("ar_pub"), "dictionary tone vs the publication-aligned AR",
        f"{S8} readers.per_reader.dictionary.ar_pub")
    est(m, "exEeightFinbertPlacebo", fb.get("placebo"), "FinBERT tone vs the placebo AR[-5,-1]",
        f"{S8} readers.per_reader.finbert.placebo")
    est(m, "exEeightDictPlacebo", dc.get("placebo"), "dictionary tone vs the placebo AR[-5,-1]",
        f"{S8} readers.per_reader.dictionary.placebo")
    for key, tag in (("emu", "Emu"), ("finbert", "Finbert"), ("dictionary", "Dict")):
        est(m, f"exEeight{tag}MainMinusPlacebo", (pr.get(key) or {}).get("main_minus_placebo"),
            f"{key}: paired weekly AR[0,+1] minus placebo", f"{S8} readers.per_reader.{key}.main_minus_placebo")
    for pair, tag in (("emu_minus_finbert", "EmuFinbert"), ("emu_minus_dictionary", "EmuDict")):
        d = pdiff.get(pair) or {}
        for crit, ctag in (("ar_0_1", "Diff"), ("placebo", "PlaceboDiff"), ("ar_pub", "PubDiff")):
            x = d.get(crit) or {}
            m.append((f"exEeight{tag}{ctag}", n(x.get("mean"), 3, sign=True), f"paired weekly {pair}, {crit}",
                      f"{S8} readers.paired_differences.{pair}.{crit}"))
            m.append((f"exEeight{tag}{ctag}CI", c(x.get("ci")), f"its NW(4) 95% interval",
                      f"{S8} readers.paired_differences.{pair}.{crit}"))
    sc = g("timestamps", "sec_clock") or {}
    m += [("exEeightSecEarningsUnits", count(sc.get("earnings_units")), "earnings-type frame units",
           f"{S8} timestamps.sec_clock.earnings_units"),
          ("exEeightSecUnits", count(sc.get("units_with_2_02_in_window")), "of them with an Item 2.02 8-K in the "
           "label window", f"{S8} timestamps.sec_clock.units_with_2_02_in_window"),
          ("exEeightSecAfterPct", n(sc.get("share_created_after_acceptance"), 0, 100), "% created after the 8-K "
           "acceptance", f"{S8} timestamps.sec_clock.share_created_after_acceptance"),
          ("exEeightSecMedianMin", n(sc.get("median_gap_minutes_signed"), 0), "median signed gap, item minus filing "
           "(minutes)", f"{S8} timestamps.sec_clock.median_gap_minutes_signed")]
    ed = g("timestamps", "edits") or {}
    m += [("exEeightEditOnePct", n(ed.get("share_edited_gt_1_min"), 1, 100), "% of frame items edited more than 1 "
           "minute after creation", f"{S8} timestamps.edits.share_edited_gt_1_min"),
          ("exEeightEditSixtyPct", n(ed.get("share_edited_gt_60_min"), 1, 100), "% edited more than 60 minutes after "
           "creation", f"{S8} timestamps.edits.share_edited_gt_60_min")]
    ac = g("timestamps", "after_close_group") or {}
    m += [("exEeightAfterNonsessionPct", n(ac.get("share_nonsession_day"), 0, 100), "% of the after-close-or-"
           "non-session group created on a non-session day", f"{S8} timestamps.after_close_group"),
          ("exEeightAfterCloseMin", n(ac.get("median_minutes_after_close"), 0), "median minutes after the close, "
           "items created after a session's close", f"{S8} timestamps.after_close_group")]
    by = g("bylines") or {}
    for key, tag in (("desk", "Desk"), ("named", "Named")):
        b = by.get(key) or {}
        m += [(f"exEeight{tag}SharePct", n(b.get("share_units"), 0, 100), f"% of frame units, byline class {key}",
               f"{S8} bylines.{key}.share_units"),
              (f"exEeight{tag}PricePct", n(b.get("price_report_share"), 1, 100), f"price-report share, {key}",
               f"{S8} bylines.{key}.price_report_share")]
        est(m, f"exEeight{tag}", b.get("ar_0_1"), f"weekly statistic vs AR[0,+1], {key}", f"{S8} bylines.{key}.ar_0_1")
        est(m, f"exEeight{tag}Pub", b.get("ar_pub"), f"weekly statistic vs the publication-aligned AR, {key}",
            f"{S8} bylines.{key}.ar_pub")
    fs = g("frame_sensitivity") or {}
    m += [("exEeightFramePairs", count(fs.get("pairs_with_effect_answer")), "item-firm pairs with an effect answer",
           f"{S8} frame_sensitivity.pairs_with_effect_answer"),
          ("exEeightFrameKeptPairs", count(fs.get("kept_pairs")), "kept pairs of every type",
           f"{S8} frame_sensitivity.kept_pairs")]
    est(m, "exEeightFrameAll", fs.get("all_with_effect"), "registered statistic on every pair with an effect answer",
        f"{S8} frame_sensitivity.all_with_effect")
    est(m, "exEeightFrameKept", fs.get("kept_all_types"), "the same on kept pairs of every type",
        f"{S8} frame_sensitivity.kept_all_types")
    est(m, "exEeightFrameAllNoPrice", fs.get("all_with_effect_no_price"), "every pair with an effect answer, without "
        "price-report items", f"{S8} frame_sensitivity.all_with_effect_no_price")
    return m


def macros_e2(e2: dict | None) -> list:
    m: list = []
    fresh = V.get(e2, "item_level_by_timing", "d0_on_decision_day") or {}
    m.append(("exEtwoFreshSharePct", n(V.get(e2, "timing", "share_d0_on_decision_day"), 0, 100), "% of frame items "
              "whose event session is the decision session (the freshest items at the fill)",
              f"{S2} timing.share_d0_on_decision_day"))
    est(m, "exEtwoFreshTraded", fresh.get("traded_holding_week"), "those items vs the traded week's relative return",
        f"{S2} item_level_by_timing.d0_on_decision_day.traded_holding_week")
    return m


def analysis_ids() -> list[str]:
    """Distinct exploratory analyses with an output in results/exploratory/ (E1-E8, PD1x, PD3x, ...)."""
    ids = set()
    for p in sorted(V.EXPLORATORY.glob("*.json")):
        mt = re.match(r"(e\d+|pd\d+)_", p.name)
        if not mt:
            continue
        meta = json.loads(p.read_text()).get("meta") or {}
        if meta.get("exploratory") is True and meta.get("label") == LABEL:
            ids.add(mt.group(1))
    return sorted(ids, key=lambda s: (s[0] != "e", int(re.sub(r"\D", "", s))))


def render() -> tuple[str, list[str]]:
    e8, e2 = V.exploratory("e8_timing_robustness"), V.exploratory("e2_priced_horizon")
    ids = analysis_ids()
    rows = macros_e8(e8) + macros_e2(e2) + [
        ("exNAnalyses", str(len(ids)) if ids else None, "exploratory analyses with an output: " + ", ".join(ids),
         "results/exploratory/*.json (meta.exploratory)")]
    lines = ["% paper/generated/exploratory_numbers_r2.tex -- GENERATED by experiments/exploratory/r2_make_numbers.py.",
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
        print("exploratory_numbers_r2: " + ("stale" if stale else "up to date") +
              (f"; missing: {', '.join(missing)}" if missing else ""))
        return 1 if stale or missing else 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}" + (f"; MISSING (placeholders): {', '.join(missing)}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
