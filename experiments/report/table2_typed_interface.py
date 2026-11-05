#!/usr/bin/env python3
"""Table 2: what the typed interface buys, every reader in the same untrusted slot, band and cap (outline_v1 §3
Table 2; design_v2_delta RB-16; the draft's caption in paper/sections/results.tex).

    .venv/bin/python experiments/report/table2_typed_interface.py   # results/analysis/{h1,frontier,pd0,
                                                                     #   localization,rta_ng}.json
    .venv/bin/python experiments/report/table2_typed_interface.py --in-dir results/analysis/W_pre_slice --out-dir X

Rows (registered readers; "---" = not run, "n/a" = not applicable): typed programs (ours); typed, no defenses; holistic;
holistic + guards; holistic + marking; typed, median aggregation; typed, majority vote (DEV-11); dictionary (LM, standing in for the cut headline-tone
reader); TA-style pipeline and Mono-CoT (S30 names, dagger); code-only.
Columns: clean selection return of the signal book at the operating point (bps per week; frontier.json
`clean_by_reader`); static in-band ASR pooled over A1-A7 with its 95% CI clustered on the attack week (h1.json
`pooled`, promotion); max |dL-bar^W| / 2 kappa over attacked units (h1.json `by_reader`, the P4 tightness measure);
signed harm of the system book per attacked unit, mean bps of NAV with CI (RB-4); localization (%) of successful
attacks (localization.json, experiments/analysis/localization.py), "n/a" for readers without a per-item channel;
invalid or degraded outputs (%), requests and $ per stock-week and p95 latency (pd0.json).
The paid adaptive attack on the emulator is cut (prereg §10), so the table has no adaptive column for the emu
readers. RT-A-NG, the adaptive attacker on the open typed model, is item level only (no portfolio replay) and is
reported in the table's note from results/analysis/rta_ng.json (experiments/analysis/rta_ng_summary.py): the share of
items whose best candidate survives N0 and the guards, and the mean E[effect] shift of surviving candidates, at 1, 4,
16 and 64 queries. Every input goes through `check_inputs` (a development file never reaches paper/tables).
The H1 statistic itself is in the caption (make_numbers: hOneDelta, hOneCI, hOnePholm).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _latex as L

NAME = "table2_typed_interface"
# (label, h1 reader, frontier clean key, pd0 comparator key or "typed", per-item channel)
READERS = (("Typed programs (ours)", "typed", "typed", "typed", True),
           ("Typed, no defenses", "nodef", None, None, True),
           ("Holistic", "H", "H", "H", False),
           ("Holistic + guards", "H-guard", "H-guard", "H-guard", False),
           ("Holistic + marking", "H-spot", "H-spot", "H-spot", False),
           ("Typed, median aggregation", "R-med", "R-med", "typed", True),
           # DEV-11 (audit A9): R-vote is a registered C1-secondary reader (prereg §3.1); its row was missing
           ("Typed, majority vote", "R-vote", "R-vote", "typed", True),
           ("Dictionary (LM)", "LM", "LM", None, False),
           ("TA-style pipeline$^\\dagger$", "TA-lite", None, "TA-lite", False),
           ("Mono-CoT$^\\dagger$", "Mono-CoT", None, "Mono-CoT", False),
           ("Code-only", None, "code-only", None, False))


def pooled(h1: dict | None, reader: str | None) -> dict | None:
    if not h1 or reader is None:
        return None
    return next((p for p in h1.get("pooled", []) if p["reader"] == reader and p["kind"] == "promotion"), None)


def rta_note(rta: dict | None) -> str:
    if not rta or not rta.get("budgets"):
        return "Adaptive attacker on the open typed model (RT-A-NG): not run."
    b = rta["budgets"]
    surv = " / ".join(L.pct(L.get(b, k, "survival_rate"), 0) for k in ("1", "4", "16", "64"))
    shift = " / ".join(L.num(L.get(b, k, "mean_delta_expected_survivors"), 2) for k in ("1", "4", "16", "64"))
    return (f"Adaptive attacker on the open typed model (RT-A-NG, {rta.get('n_items')} items, item level): best "
            f"candidate survives N0 and the guards in {surv}\\% of items after 1 / 4 / 16 / 64 queries; mean "
            f"E[effect] shift of surviving candidates {shift} levels.")


def build(h1: dict | None, fr: dict | None, pd0: dict | None, loc: dict | None, rta: dict | None,
          aliases: dict | None = None) -> str:
    aliases = aliases or {}
    body = []
    for label, r1, rc, r0, per_item in READERS:
        r1 = aliases.get(r1, r1)
        clean = L.get(fr, "clean_by_reader", rc) if rc else None
        sel = L.num(clean.get("selection_return_bps"), 1) if clean and clean.get("status") == "ok" else L.NOT_RUN
        p = pooled(h1, r1)
        asr = L.est_ci(p.get("asr"), p.get("ci"), 1, 100.0) if p and p.get("asr") is not None else L.NOT_RUN
        br = L.get(h1, "by_reader", r1) if r1 else None
        dl = L.num(L.get(br, "dL_over_2kappa", "max"), 2) if br else L.NOT_RUN
        harm = L.est_ci(L.get(br, "harm_sized", "mean_bps"), L.get(br, "harm_sized", "mean_ci"), 1) if br \
            else L.NOT_RUN
        if not per_item:
            locz = L.NA
        elif loc and L.get(loc, "by_reader", r1, "status") == "ok":
            locz = L.pct(loc["by_reader"][r1].get("rate"), 1)
        else:
            locz = L.NOT_RUN
        if r0 == "typed":
            prof = L.get(pd0, "typed", "total")
            inv = L.get(prof, "degraded_share")
        elif r0:
            comp = L.get(pd0, "comparators", r0)
            prof = L.get(comp, "total") if comp and comp.get("status") != "not run" else None
            inv = L.get(comp, "invalid_share", "invalid_final") if comp and "invalid_share" in comp \
                else L.get(prof, "degraded_share")
        else:
            prof, inv = None, None
        body.append(L.row(label, sel, asr, dl, harm, locz, L.pct(inv, 2) if prof else L.NOT_RUN,
                          L.num(L.get(prof, "requests_per_stock_week"), 1),
                          L.num(L.get(prof, "usd_per_stock_week"), 3, 1e3),
                          L.num(L.get(prof, "latency_p95_s"), 2)))
    header = ("Reader & Clean sel. (bps/wk) & Static ASR (\\%) & max $|\\Delta \\bar L^W|/2\\kappa$ & "
              "Harm (bps) & Local. (\\%) & Invalid (\\%) & Req./sw & \\$/sw ($\\times10^{-3}$) & p95 (s) \\\\")
    c1 = L.get(h1, "confirmatory") or {}
    notes = []
    if c1.get("status") == "ok":
        notes.append(f"Paired pairs {c1.get('n_pairs')} in {c1.get('n_clusters')} attack weeks; excluded (typed only / "
                     f"other only / both) {c1.get('n_excluded_a_only')} / {c1.get('n_excluded_b_only')} / "
                     f"{c1.get('n_excluded_both')}.")
    notes.append(rta_note(rta))
    return L.tabular("lrrrrrrrrr", header, body, " ".join(notes), size="\\scriptsize")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alias", action="append", default=[], help="registered=label, e.g. H=R-med (development runs)")
    a = ap.parse_args(argv)
    h1, fr, pd0 = L.load("h1", a.in_dir), L.load("frontier", a.in_dir), L.load("pd0", a.in_dir)
    loc, rta = L.load("localization", a.in_dir), L.load("rta_ng", a.in_dir)
    out = L.check_inputs([h1, fr, pd0, loc, rta], a.out_dir)
    p = L.write(out, NAME, build(h1, fr, pd0, loc, rta, dict(x.split("=", 1) for x in a.alias)))
    print(f"table2 -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
