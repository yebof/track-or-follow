#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the \\exEnine macros of E9 (does the information survive at the source? the
decomposition of AR[0,+1] around the 8-K acceptance and the reader's association with the return from the first trade
after acceptance vs after publication) -> paper/generated/exploratory_numbers_e9.tex.

    .venv/bin/python experiments/exploratory/e9_make_numbers.py [--check]

Every macro name starts with \\exEnine and every comment carries the label "exploratory (not pre-registered)"; a
sentence that uses one says "exploratory" in the same sentence (outline_v2 §5.5). The input comes only through the
exploratory loader (experiments/report/_validity.py: meta.exploratory true, the label, the lock open). A missing input
leaves a bold [name] placeholder and exit 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402  (exploratory loader)
from make_numbers import fmt_num  # noqa: E402  (read-only)
from r1_make_numbers import c, est, n  # noqa: E402  (same formatting as the round-1 macros)

OUT = ROOT / "paper" / "generated" / "exploratory_numbers_e9.tex"
LABEL = V.LABEL_X
S9 = "results/exploratory/e9_source_timing.json"
GROUPS = (("earnings", "Earn"), ("other", "Other"), ("all", "All"))
COMPONENTS = (("pre", "Pre"), ("between", "Between"), ("after", "After"))
CRITERIA = (("ar_0_1", "Label"), ("ar_acc", "Acc"), ("ar_pub", "Pub"), ("ar_acc_regular", "AccRegular"),
            ("acc_minus_pub", "Diff"))


def count(x) -> str | None:
    return fmt_num(int(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def macros(e9: dict | None) -> list:
    m: list = []
    for key, tag in GROUPS:
        g = V.get(e9, "groups", key) or {}
        src = f"{S9} groups.{key}"
        ct = g.get("counts") or {}
        m += [(f"exEnine{tag}Matched", count(ct.get("matched")), f"{key}: frame units with a matched 8-K",
               f"{src}.counts.matched"),
              (f"exEnine{tag}Units", count(ct.get("decomposition")), f"{key}: units in the decomposition (filing "
               "first, inside the AR[0,+1] window, all returns finite)", f"{src}.counts.decomposition"),
              (f"exEnine{tag}Events", count(ct.get("decomposition_events")), f"{key}: distinct (firm, 8-K) events "
               "in the decomposition", f"{src}.counts.decomposition_events"),
              (f"exEnine{tag}NewsFirst", count(ct.get("news_first")), f"{key}: matched units published before "
               "the acceptance", f"{src}.counts.news_first"),
              (f"exEnine{tag}FilingFirstPct", n(g.get("share_filing_first_among_matched"), 0, 100),
               f"{key}: % of matched units published after the acceptance",
               f"{src}.share_filing_first_among_matched"),
              (f"exEnine{tag}GapMedianMin", n(V.get(g, "minutes_acceptance_to_publication", "median"), 0),
               f"{key}: median minutes acceptance -> publication (decomposition units)",
               f"{src}.minutes_acceptance_to_publication.median"),
              (f"exEnine{tag}EntryLagMin", n(V.get(g, "minutes_acceptance_to_entry_bar", "median"), 1),
               f"{key}: median minutes acceptance -> first one-minute bar",
               f"{src}.minutes_acceptance_to_entry_bar.median"),
              (f"exEnine{tag}AfterClosePct", n(V.get(g, "acceptance_clock", "after_close", "share"), 0, 100),
               f"{key}: % of decomposition units accepted after the close (16:00-20:00 ET)",
               f"{src}.acceptance_clock.after_close.share"),
              (f"exEnine{tag}PreOpenPct", n(V.get(g, "acceptance_clock", "pre_open", "share"), 0, 100),
               f"{key}: % accepted before the open", f"{src}.acceptance_clock.pre_open.share")]
        for level, ltag in (("unit_level", ""), ("event_level", "Ev")):
            d = V.get(g, "decomposition", level) or {}
            for comp, ctag in COMPONENTS:
                x = d.get(comp) or {}
                m += [(f"exEnine{tag}{ltag}{ctag}Pct", n(x.get("sign_aligned"), 0, 100),
                       f"{key}, {level}: % of the sign-aligned AR[0,+1] realized in component {comp}",
                       f"{src}.decomposition.{level}.{comp}.sign_aligned"),
                      (f"exEnine{tag}{ltag}{ctag}PctCI", c(x.get("sign_aligned_ci"), 0, 100),
                       "its week-block bootstrap 95% interval (%)", f"{src}.decomposition.{level}.{comp}"),
                      (f"exEnine{tag}{ltag}{ctag}Slope", n(x.get("slope"), 2),
                       f"{key}, {level}: OLS slope of component {comp} on AR[0,+1]",
                       f"{src}.decomposition.{level}.{comp}.slope"),
                      (f"exEnine{tag}{ltag}{ctag}SlopeCI", c(x.get("slope_ci"), 2),
                       "its week-block bootstrap 95% interval", f"{src}.decomposition.{level}.{comp}")]
            a = V.get(g, "association", level) or {}
            for crit, rtag in CRITERIA:
                est(m, f"exEnine{tag}{ltag}Rho{rtag}", a.get(crit), f"{key}, {level}: weekly Spearman of E[effect] "
                    f"with {crit} (NW(4) 95% interval)", f"{src}.association.{level}.{crit}")
        es = g.get("earliest_source_on_matched") or {}
        est(m, f"exEnine{tag}RhoSrc", es.get("ar_src"), f"{key}: weekly Spearman with the return from the first "
            "trade after the earlier of acceptance and publication, all matched units",
            f"{src}.earliest_source_on_matched.ar_src")
        est(m, f"exEnine{tag}RhoSrcPub", es.get("ar_pub"), f"{key}: the same with the publication-aligned return",
            f"{src}.earliest_source_on_matched.ar_pub")
        est(m, f"exEnine{tag}RhoSrcDiff", es.get("src_minus_pub"), f"{key}: paired weekly difference of the two",
            f"{src}.earliest_source_on_matched.src_minus_pub")
    m.append(("exEnineCacheDates", count(V.get(e9, "minute_bars", "session_dates")), "session dates in the minute-bar "
              "cache (SIP, 04:00-20:00 ET)", f"{S9} minute_bars.session_dates"))
    return m


def render() -> tuple[str, list[str]]:
    rows = macros(V.exploratory("e9_source_timing"))
    lines = ["% paper/generated/exploratory_numbers_e9.tex -- GENERATED by experiments/exploratory/e9_make_numbers.py.",
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
        print("exploratory_numbers_e9: " + ("stale" if stale else "up to date") +
              (f"; missing: {', '.join(missing)}" if missing else ""))
        return 1 if stale or missing else 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}" + (f"; MISSING (placeholders): {', '.join(missing)}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
