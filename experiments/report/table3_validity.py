#!/usr/bin/env python3
"""Table 3: validity, signal and composition (outline_v1 §3 Table 3 as cut by the draft-0 review: panel A = H2 per
implementation, panel B = H3 on both windows with its split, composition and the all-stores-off row). Writes
paper/tables/table3_validity_signal.tex (the name paper/sections/results.tex inputs).

    .venv/bin/python experiments/report/table3_validity.py         # results/analysis/{h2,h3,pd2,pd3,confirmatory}.json
    .venv/bin/python experiments/report/table3_validity.py --in-dir results/analysis/W_pre_slice --out-dir X

Panel A (columns emu | VERB | NG; a cell without its statistic shows "---"): Spearman of E[effect] with AR[0,+1] in
two groups of rows (DEV-11): the weekly mean with its NW 95% CI (emu's confirmatory statistic, with items created
before 09:30 ET, placebo AR[-5,-1] and W_release), then the pooled Spearman with week-block CIs (emu on its full
frame, VERB and NG on their samples, descriptive, with the pooled placebo); AUC of P(event type) against 8-K
Items 2.02, 1.01/2.01 and 5.02; AUC of vs_expectations against the sign of AR; W_release; units.
Panel B: H3 dIC with its CI and the Holm-adjusted p (confirmatory.json), on W_clean and W_release; the split into
trusted typed judgments and untrusted text; the increment over the coverage placebo; composition: IC of the learned
pool, the equal-weight log pool and the best single program, the learned pool's oracle-gap share; the
all-stores-off dIC (secondary, PD3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _latex as L

NAME = "table3_validity_signal"
IMPLS = ("emu", "VERB", "NG")
SEC_ROWS = (("earnings_results_vs_2.02", "AUC, earnings vs 8-K 2.02"),
            ("deal_or_investment_vs_1.01/2.01", "AUC, deal vs 8-K 1.01/2.01"),
            ("management_or_labor_vs_5.02", "AUC, management vs 8-K 5.02"))


def wk(x: dict | None, d: int = 3) -> str:
    return L.est_ci(x.get("mean"), x.get("ci"), d) if x else L.NOT_RUN


def pooled(x: dict | None, d: int = 3) -> str:
    return L.est_ci(x.get("rho"), x.get("ci_week_block"), d) if x else L.NOT_RUN


def panel_a(h2: dict | None) -> list[str]:
    impl = {k: L.get(h2, "per_implementation", k) or {} for k in IMPLS}
    ok = {k: impl[k].get("status") == "ok" for k in IMPLS}

    def cells(emu: str, other) -> list[str]:
        return [emu] + [other(impl[k]) if ok[k] else L.NOT_RUN for k in IMPLS[1:]]

    # DEV-11 (audit A6): emu's confirmatory statistic is the WEEKLY mean (NW CI); VERB and NG report the POOLED
    # Spearman (week-block CI). Each statistic has its own group of rows, so that no row compares different statistics.
    rows = [L.row("Spearman, AR$[0,+1]$, weekly mean", wk(L.get(h2, "confirmatory")), L.NOT_RUN, L.NOT_RUN),
            L.row("\\quad items before 09:30 ET", wk(L.get(h2, "controls", "pre_open_only")), L.NOT_RUN, L.NOT_RUN),
            L.row("\\quad placebo AR$[-5,-1]$", wk(L.get(h2, "placebo_ar_m5_m1")), L.NOT_RUN, L.NOT_RUN),
            L.row("\\quad $W_{\\mathrm{release}}$", wk(L.get(h2, "sensitivities", "W_release")), L.NOT_RUN,
                  L.NOT_RUN),
            L.row("Spearman, AR$[0,+1]$, pooled", pooled(L.get(h2, "per_implementation", "emu", "spearman_ar_pooled")),
                  *[pooled(impl[k].get("spearman_ar")) if ok[k] else L.NOT_RUN for k in IMPLS[1:]]),
            L.row("\\quad placebo AR$[-5,-1]$", L.NOT_RUN,
                  *[pooled(impl[k].get("spearman_ar_placebo")) if ok[k] else L.NOT_RUN for k in IMPLS[1:]])]
    for key, lab in SEC_ROWS:
        rows.append(L.row(lab, *cells(L.num(L.get(h2, "sec_8k", key, "auc"), 3),
                                      lambda x, k=key: L.num(L.get(x, "sec_8k", k, "auc"), 3))))
    rows.append(L.row("AUC, vs.\\ expectations vs sign of AR",
                      *cells(L.num(L.get(h2, "vs_expectations_vs_ar_sign", "auc"), 3),
                             lambda x: L.num(L.get(x, "vs_expectations_vs_ar_sign", "auc"), 3))))
    n_emu = L.get(h2, "confirmatory", "n_units")
    rows.append(L.row("Units (weeks)", f"{n_emu:,}".replace(",", "{,}") + f" ({L.get(h2, 'confirmatory', 'n_weeks')})"
                      if n_emu else L.NOT_RUN,
                      *[f"{impl[k].get('n_units'):,}".replace(",", "{,}") if ok[k] else L.NOT_RUN
                        for k in IMPLS[1:]]))
    return rows


def panel_b(h3: dict | None, pd2: dict | None, pd3: dict | None, conf: dict | None) -> list[str]:
    fam = {r["hypothesis"]: r for r in (L.get(conf, "family") or [])}
    c = L.get(h3, "confirmatory") or {}
    rel = L.get(h3, "sensitivities", "W_release") or {}
    rows = [L.row("H3: $\\Delta$IC vs code-only", L.est_ci(c.get("mean"), c.get("ci"), 3),
                  L.est_ci(rel.get("mean"), rel.get("ci"), 3)),
            L.row("\\quad Holm-adjusted $p$", L.num(L.get(fam, "H3", "p_holm"), 3), L.NA),
            L.row("\\quad trusted typed judgments", wk(L.get(h3, "split", "trusted_typed_part_IC_LT_minus_IC_codeonly")),
                  L.NA),
            L.row("\\quad untrusted text", wk(L.get(h3, "split", "text_part_IC_L_minus_IC_LT")), L.NA),
            L.row("\\quad over the coverage placebo", wk(L.get(h3, "coverage_placebo", "delta_ic_over_placebo")),
                  L.NA)]
    for k, lab in (("primary", "IC, learned pool"), ("V-EW-log", "IC, equal-weight log pool"),
                   ("V-best", "IC, best single program")):
        rows.append(L.row(lab, wk(L.get(pd2, "pools", k, "ic")), L.NA))
    g = L.get(pd2, "oracle_gap_share", "primary") or {}
    rows.append(L.row("Oracle-gap share, learned pool", L.est_ci(g.get("share"), g.get("ci"), 2), L.NA))
    st = L.get(pd3, "delta_ic") if L.get(pd3, "status") == "ok" else None
    rows.append(L.row("All stores off: $\\Delta$IC (secondary)", wk(st), L.NA))
    return rows


def build(h2, h3, pd2, pd3, conf) -> str:
    a = panel_a(h2)
    b = panel_b(h3, pd2, pd3, conf)
    body = ["\\multicolumn{4}{l}{\\textsc{a. h2 per implementation}} \\\\", *a, "\\midrule",
            "\\multicolumn{4}{l}{\\textsc{b. signal and composition}} \\\\",
            L.row("", "$W_{\\mathrm{clean}}$", "$W_{\\mathrm{release}}$", ""),
            *[r.replace(" \\\\", " & \\\\") for r in b]]
    header = "& emu & VERB & NG \\\\"
    return L.tabular("lrrr", header, body, size="\\footnotesize")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    docs = [L.load(n, a.in_dir) for n in ("h2", "h3", "pd2", "pd3", "confirmatory")]
    out = L.check_inputs(docs, a.out_dir)
    p = L.write(out, NAME, build(*docs))
    print(f"table3 -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
