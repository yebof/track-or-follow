#!/usr/bin/env python3
"""Table 1 of the validity-first paper: validity of typed item judgments, by reader (outline_v2 §3 "Table 1"; DEV-12).

    .venv/bin/python experiments/report/table1_validity.py      # -> paper/tables/table1_validity.tex

Columns: status (C = confirmatory, R = registered, X = exploratory) | measure | emu (the confirmatory reader) | VERB
(a verbalized contrast) | NG (an open non-generative model) [| Jev: a column appears only if a registered Jev row
exists, experiments/report/_validity.py]. Cells: estimate with its 95% interval below it in small type; "---" not
computed; "n/a" not applicable; counts in parentheses.

Panel A, market reactions (Spearman correlation of E[effect] with the abnormal return): the confirmatory weekly mean
over W_clean and its registered controls and sensitivities (emu only; the descriptive readers have about 4.5 units per
week, too few for weekly means), then the pooled statistic of every reader with its placebo, emu on the same units,
and the agreement with emu. Panel B, concepts vs the firm's own 8-K item codes: AUC [CI] (positives), the
vs_expectations AUC (n) and the ECE of the earnings concept. Panel C, samples and cost.

Inputs (registered, lock open; refused otherwise): results/analysis/h2.json, confirmatory.json, pd0.json;
results/prereg/power_prereg.json (NG CPU seconds); VERB's MANIFEST.json (paid spend of record). Writes only the
tabular; the caption and label live in paper/sections/. No p-value appears in the table (the Holm-adjusted p of H2 is
the caption's \\hTwoPholm).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _latex as L  # noqa: E402  (registered formatting helpers, read-only)
import _xtab as XT  # noqa: E402  (tabularx for alternating row colours)
import _validity as V  # noqa: E402

NAME = "table1_validity"
ROOT = V.ROOT
VERB_MANIFEST = ROOT / "results/judgments/verb_llm-verbalized_wclean_items2000/MANIFEST.json"
POWER = ROOT / "results/prereg/power_prereg.json"


def est(x, d=3) -> str:
    return L.num(x, d) if L.finite(x) else L.NOT_RUN


def ci_(c, d=3) -> str:
    """The 95% interval in its own column at the same type size (round-2 figure review T1-2, T1-3); '---' when not
    computed."""
    return L.ci(*(c or [None, None])[:2], d) or L.NOT_RUN


def weekly(d: dict | None) -> tuple[str, str]:
    return est(V.get(d, "mean")), ci_(V.get(d, "ci"))


def pooled(d: dict | None) -> tuple[str, str]:
    return est(V.get(d, "rho")), ci_(V.get(d, "ci_week_block"))


def positives(s: dict | None) -> int | None:
    if not s or not (L.finite(s.get("prevalence")) and L.finite(s.get("n"))):
        return None
    return int(round(s["prevalence"] * s["n"]))


def fmt_int(x) -> str:
    return f"{x:,}".replace(",", "{,}") if x is not None else L.NOT_RUN


def tabular_star(colspec: str, rows: list[str]) -> list[str]:
    """A tabular* at the enclosing width, top-aligned ([t]: round-2 figure review T1-1, so side-by-side panels share
    their top rule)."""
    return [XT.begin("\\linewidth", colspec, "[t]"), "\\toprule", *rows, "\\bottomrule", XT.END]


def minipage(width: str, colspec: str, rows: list[str]) -> str:
    return "\n".join([f"\\begin{{minipage}}[t]{{{width}}}", *tabular_star(colspec, rows), "\\end{minipage}"])


def reader_rows(h2: dict, readers: list[str]) -> list[str]:
    """Pooled registered estimates of the readers on their own sampled items: an estimate and a 95% interval column per
    reader; counts in the row label, one per reader."""
    impl = h2["per_implementation"]
    ok = {r: V.get(impl, r, "status") == "ok" for r in readers}
    NR = L.NOT_RUN
    fr = h2.get("implementation_frames") or {}

    def cells(f) -> list[str]:
        out = []
        for r in readers:
            out += list(f(r)) if ok[r] else [NR, NR]
        return out

    def counts(f) -> str:
        return "; ".join(fmt_int(f(r)) if ok[r] else NR for r in readers)

    rows = [L.row("\\ARzo{}", *cells(lambda r: pooled(V.get(impl, r, "spearman_ar")))),
            L.row("\\quad emu on the same items", *cells(lambda r: pooled(V.get(impl, r, "emu_same_units")))),
            L.row("Placebo \\ARpl{}", *cells(lambda r: pooled(V.get(impl, r, "spearman_ar_placebo")))),
            L.row("Agreement with emu, $\\rho$ of \\Eeff{}",
                  *cells(lambda r: (L.num(V.get(impl, r, "agreement", "spearman_E_effect"), 2), NR))),
            L.row("Agreement with emu, $\\kappa$ of top type",
                  *cells(lambda r: (L.num(V.get(impl, r, "agreement", "top_event_type", "kappa"), 2), NR)))]
    short = {"Earnings results": "Earnings", "Deal or investment": "Deal", "Management or labor": "Management"}
    for key, label, codes in V.SEC_CONCEPTS:
        pos = counts(lambda r: positives(V.get(impl, r, "sec_8k", key)))
        rows.append(L.row(f"{short.get(label, label)} vs Item{'s' if '/' in codes else ''} {codes} ({pos} pos.)",
                          *cells(lambda r: (est(V.get(impl, r, "sec_8k", key, "auc")),
                                            ci_(V.get(impl, r, "sec_8k", key, "auc_ci"))))))
    nvs = counts(lambda r: V.get(impl, r, "vs_expectations_vs_ar_sign", "n"))
    rows.append(L.row(f"Beat-expectations vs sign ($n$ {nvs})",
                      *cells(lambda r: (est(V.get(impl, r, "vs_expectations_vs_ar_sign", "auc")), NR))))
    rows.append(L.row("ECE, earnings vs Item 2.02",
                      *cells(lambda r: (est(V.get(impl, r, "sec_8k", "earnings_results_vs_2.02", "ece")), NR))))
    rows.append(L.row("Sampled items (eligible by own answers)",
                      *cells(lambda r: (fmt_int(V.get(fr, r, "units")), f"({fmt_int(V.get(fr, r, 'c2_eligible'))})"))))
    return rows


def build(h2: dict) -> str:
    impl = h2["per_implementation"]
    jev = V.jev_key(h2)
    right = ["VERB"] + ([jev] if jev else [])
    c = h2["confirmatory"]
    st = h2.get("settings") or {}
    drop = next((k for k in (h2.get("sensitivities") or {}) if k.startswith("drop_first_")), None)
    sec = V.get(impl, "emu", "sec_8k_full_frame") or {}
    vs = h2.get("vs_expectations_vs_ar_sign") or {}
    left = ["\\multicolumn{4}{l}{\\textit{A. Confirmatory reader (emu), all %s frame items}} \\\\"
            % fmt_int(c["n_units"]),
            "\\midrule", "& Measure & Est. & 95\\% CI \\\\", "\\midrule",
            L.row("C", f"\\ARzo{{}}, weekly mean ({c['n_weeks']} weeks)", *weekly(c)),
            L.row("R", f"\\quad after the model's release ({V.get(h2, 'sensitivities', 'W_release', 'n_weeks')} weeks)",
                  *weekly(V.get(h2, "sensitivities", "W_release"))),
            L.row("R", f"\\quad published before {st.get('pre_open_before')} ET",
                  *weekly(V.get(h2, "controls", "pre_open_only"))),
            L.row("R", "\\quad declared prior looks excluded", *weekly(V.get(h2, "controls", "prior_look_excluded"))),
            L.row("R", f"\\quad first {drop.rsplit('_', 1)[-1]} weeks dropped" if drop else "\\quad drop sensitivity",
                  *weekly(V.get(h2, "sensitivities", drop))),
            L.row("R", "Placebo \\ARpl{}, weekly mean", *weekly(h2.get("placebo_ar_m5_m1"))),
            L.row("R", "\\ARzo{}, pooled over items", *pooled(V.get(impl, "emu", "spearman_ar_pooled"))),
            "\\midrule",
            "\\multicolumn{4}{l}{\\textit{8-K agreement (AUC) and calibration}} \\\\"]
    for key, label, codes in V.SEC_CONCEPTS:
        s = sec.get(key) or {}
        left.append(L.row("R", f"{label} vs Item{'s' if '/' in codes else ''} {codes} ({fmt_int(positives(s))} pos.)",
                          est(s.get("auc")), ci_(s.get("auc_ci"))))
    left.append(L.row("R", f"Beat-expectations judgment vs sign of \\ARzo{{}} ($n$ {fmt_int(vs.get('n'))})",
                      est(vs.get("auc")), L.NOT_RUN))
    left.append(L.row("R", "ECE, earnings vs Item 2.02", est(V.get(sec, "earnings_results_vs_2.02", "ece")),
                      L.NOT_RUN))
    head = " & ".join(["Measure (pooled over items)"] + [f"{V.READERS.get(r, V.READERS['Jev'])[0]} & 95\\% CI"
                                                         for r in right]) + " \\\\"
    title = ("B. Verbalized contrast (VERB) on its sampled frame items" if not jev
             else "B. VERB and Jev on their sampled frame items")
    rrows = [f"\\multicolumn{{{1 + 2 * len(right)}}}{{l}}{{\\textit{{{title}}}}} \\\\", "\\midrule", head,
             "\\midrule", *reader_rows(h2, right)]
    if jev:  # round-2 figure review J-1: with a Jev column, panel B needs the full width; stack the panels
        body = [minipage("\\linewidth", "clrl", left), "\\par\\medskip",
                minipage("\\linewidth", "l" + "rl" * len(right), rrows)]
    else:
        body = [minipage("0.515\\linewidth", "clrl", left), "\\hfill",
                minipage("0.465\\linewidth", "l" + "rl" * len(right), rrows)]
    lines = ["% GENERATED by experiments/report/table1_validity.py -- do not edit by hand (CLAUDE.md rule 2).",
             "{\\footnotesize", "\\setlength{\\tabcolsep}{5pt}", *body,
             "\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\scriptsize\\raggedright",
             ("C: confirmatory (one-sided, Holm over H1--H3); R: registered estimate, 95\\% interval, no test; "
              "---: not computed; pos.: positives. VERB: the same LLM's verbalized probabilities on a seeded sample of "
              "frame items, pooled over items. NG: Table~\\ref{tab:ng}."
              + (" Jev's rows are \\jevPanelStatus{}s on its own sampled items, \\jevPanelTiming{}." if jev else "")),
             "\\end{minipage}", "}"]
    return "\n".join(lines) + "\n"


def build_ng(h2: dict) -> str:
    """App. C table: the open non-generative model's registered rows (moved out of the body; PC review M3)."""
    rows = ["Measure (pooled over items) & NG & 95\\% CI \\\\", "\\midrule", *reader_rows(h2, ["NG"])]
    # natural width (round-2 figure review A-5: no 40 mm gap between labels and values)
    return V.tabular("lrl", rows, None, size="\\scriptsize", width=None, sep="3pt",
                     source="experiments/report/table1_validity.py")


def main(argv=None) -> int:
    h2 = V.require_registered("h2")
    V.require_registered("confirmatory")
    p = L.write(L.TABLE_DIR, NAME, build(h2))
    q = L.write(L.TABLE_DIR, "tableC_ng", build_ng(h2))
    print(f"table1_validity -> {p.relative_to(ROOT)}; {q.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
