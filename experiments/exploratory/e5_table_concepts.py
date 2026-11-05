#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E5: Table 2 of the validity-first paper, where the reader tracks the reaction
(outline_v2 §3 "Table 2"; the paper's error analysis).

    .venv/bin/python experiments/exploratory/e5_table_concepts.py    # -> paper/tables/table2_concepts.tex

Rows: every top event type of the confirmatory reader in the H2 frame, in frame-share order. Columns: share of the frame
(%); the registered weekly statistic (mean over weeks of the Spearman correlation of E[effect] with AR[0,+1]) within
the type, with its two-sided Newey-West(4) 95% interval (X); the same without items whose text reports the price move
(X); the publication-aligned statistic of E6 (X; results/exploratory/e6_publication_timing.json); and, where the
type has an 8-K item code, the registered concept AUC of the confirmatory reader against the firm's
8-K filings (R; results/analysis/h2.json sec_8k). The exploratory columns come only through the exploratory loader
(results/exploratory/e5_construct_checks.json); the registered column only through the registered guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "report"))
import _latex as L  # noqa: E402  (registered formatting helpers, read-only)
import _validity as V  # noqa: E402

NAME = "table2_concepts"


MIN_SHARE = 0.05  # rows: types with at least 5% of the frame (round-1 editor review C6)


def wk(d: dict | None) -> str:
    """A weekly estimate, marked with a dagger when its 95% interval excludes zero (round-2 figure review T2-1: a symbol,
    not italics, which cannot be seen at print size; the dagger hangs right so the digits stay aligned)."""
    if not d or not L.finite(d.get("mean")):
        return L.NOT_RUN
    v = L.num(d["mean"], 3)
    c = d.get("ci") or [None, None]
    excludes = L.finite(c[0]) and L.finite(c[1]) and (c[0] > 0 or c[1] < 0)
    return v + ("\\rlap{$^{\\dagger}$}" if excludes else "")


SHORT = {"industry_or_macro_readthrough": "Industry read-through", "other": "Other development",
         "operations_disruption": "Operations disruption"}


def build(e5: dict, e6: dict | None, h2: dict) -> str:
    bc = e5["by_concept"]
    pub = ((e6 or {}).get("publication_aligned") or {}).get("by_type") or {}
    order = [k for k in sorted(bc, key=lambda k: -bc[k]["share"]) if bc[k]["share"] >= MIN_SHARE]
    rest = [k for k in bc if k not in order]
    sec = h2.get("sec_8k") or {}
    body = []
    for tt in order:
        v = bc[tt]
        key = V.TYPE_TO_SEC.get(tt)
        s = sec.get(key) if key else None
        auc = L.num(s["auc"], 3) if s else L.NA
        body.append(L.row(SHORT.get(tt, V.EVENT_TYPE_LABEL.get(tt, tt.replace("_", " "))), L.pct(v["share"], 1),
                          wk(v["all"]), wk(v["no_price"]), wk((pub.get(tt) or {}).get("ar_pub")), auc))
    codes = "; ".join(f"{lab.lower()}: {c}" for _, lab, c in V.SEC_CONCEPTS)
    w = e5["weekly"]
    pa = (e6 or {}).get("publication_aligned") or {}
    body += ["\\midrule", L.row("All types", L.pct(sum(v["share"] for v in bc.values()), 1), wk(w["all"]),
                                 wk(w["no_price"]), wk(pa.get("all")), L.NA)]
    rows = ["& & \\multicolumn{3}{c}{Exploratory (X): weekly $\\rho$} & (R) \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(l){6-6}",
            "Reader's top & Share & \\multicolumn{2}{c}{\\ARzo{}} & From & 8-K \\\\",
            "\\cmidrule(lr){3-4}",
            "event type & (\\%) & all & no price & publ. & AUC \\\\",
            "\\midrule", *body]
    small = ", ".join(V.EVENT_TYPE_LABEL.get(k, k).lower() for k in sorted(rest, key=lambda k: -bc[k]["share"]))
    note = (f"\\textbf{{Exploratory}}, except the registered 8-K column. $\\rho$: weekly mean Spearman correlation of "
            f"\\Eeff{{}} with the abnormal return; $^{{\\dagger}}$: 95\\% interval excludes zero. No price: items that do "
            f"not report the price move. From publ.: from the first trade after publication to the next close. Types "
            f"below {100 * MIN_SHARE:.0f}\\% of the frame are omitted.")
    return V.tabular("@{}lrrrrr@{}", rows, note, size="\\scriptsize", width="\\linewidth", sep="2pt",
                     source=f"experiments/exploratory/e5_table_concepts.py ({V.LABEL_X}, except the 8-K column)")


def main(argv=None) -> int:
    e5 = V.exploratory("e5_construct_checks")
    if e5 is None:
        raise SystemExit("run experiments/exploratory/e5_construct_checks.py first")
    h2 = V.require_registered("h2")
    text = build(e5, V.exploratory("e6_publication_timing"), h2)
    p = L.write(L.TABLE_DIR, NAME, text)
    print(f"table2_concepts -> {p.relative_to(ROOT)} ({V.LABEL_X})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
