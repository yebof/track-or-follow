#!/usr/bin/env python3
"""Table 3 of the validity-first paper: planted content and the bound (outline_v2 §3 "Table 3"; DEV-12).

    .venv/bin/python experiments/report/table3_planted.py     # -> paper/tables/table3_planted.tex

Upper block (registered): static attack success (ASR, the attacked name enters the signal book) of the confirmatory
typed reader per attack family A1-A7, with its attack-week-clustered 95% interval and the number of attacked units;
then the adaptive attacker on the open typed model (RT-A-NG, item level: surviving edit and one-level shift at the
largest query budget). Lower block: every reader in the same untrusted slot, band and cap, pooled over A1-A7: ASR [CI],
the paired difference typed - reader in percentage points [CI] with its Holm-adjusted p (H1 for the holistic reader,
status C; the seven registered secondary contrasts otherwise, status R), and the share of successes localized to the
planted item (n/a for readers without a per-item channel). No TOST parity column (App. E). Footer (registered): the
bound check and its tightness witnesses.

Inputs (registered, lock open; refused otherwise): results/analysis/h1.json, confirmatory.json, localization.json,
bound.json, rta_ng.json; K and J from configs/system_v1.yaml. Writes only the tabular; caption and label live in
paper/sections/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _latex as L  # noqa: E402  (registered formatting helpers, read-only)
import _xtab as XT  # noqa: E402  (tabularx for alternating row colours)
import _validity as V  # noqa: E402

NAME = "table3_planted"
ROOT = V.ROOT
FAMILIES = (("A1", "instruction to AI readers"), ("A2", "fabricated fact"),
            ("A3", "self-labelled good news"), ("A4", "hidden instruction"), ("A5", "stock promotion"),
            ("A6", "planted press release"), ("A7", "repeated plant"))
READERS = (("typed", "Per-item reader", True), ("nodef", "\\quad no defenses", True),
           ("H", "Holistic reader (H1)", False), ("H-guard", "\\quad with guard questions", False),
           ("H-spot", "\\quad with marking", False), ("R-med", "Per-item, median", True),
           ("R-vote", "Per-item, majority vote", True), ("LM", "Dictionary reader", False))
# TA-lite (the TA-style pipeline's contrast) is registered but was not run: named in the note, no all-dash row
# (editor review B4).


def ci_col(lo, hi, d=1, scale=100.0) -> str:
    return L.ci(lo, hi, d, scale) or L.NOT_RUN


def cp(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval; used for zero or near-zero counts, where the percentile cluster bootstrap
    collapses (DEV-13; stats review M6)."""
    from scipy.stats import beta

    a = 1 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def asr_cells(r: dict | None) -> tuple[str, str]:
    """ASR (%) and its interval in two columns: the registered attack-week-clustered interval, or the exact interval
    when at most one unit succeeded (marked *, hung right so the brackets stay aligned: round-2 figure review T4-3)."""
    if not r or not L.finite(r.get("asr")):
        return L.NOT_RUN, L.NOT_RUN
    n = int(r["n"])
    k = int(round(r["asr"] * n))
    if k <= 1:
        lo, hi = cp(k, n)
        return L.num(r["asr"], 1, 100.0), ci_col(lo, hi) + "\\rlap{$^{*}$}"
    return L.num(r["asr"], 1, 100.0), ci_col(*(r.get("ci") or [None, None]))


def p_fmt(p) -> str:
    if not L.finite(p):
        return L.NOT_RUN
    return "$<$0.001" if p < 0.001 else ("$>$0.99" if p > 0.99 else L.num(p, 3))


def tab(colspec: str, rows: list[str]) -> list[str]:
    return [XT.begin("\\linewidth", colspec), "\\toprule", *rows, "\\bottomrule", XT.END]


def build(h1: dict, conf: dict, loc: dict, bound: dict, rta: dict) -> str:
    """Two panels with their own headers (round-2 figure review T4-1); no blank cells: '---' not run, 'ref.' the
    reference row of the difference, 'n/a' not applicable (T4-2); the adaptive attacker in the note (T4-4)."""
    fam = {r["family"]: r for r in h1["per_family"] if r["kind"] == "promotion" and r["reader"] == "typed"}
    nod = {r["family"]: r for r in h1["per_family"] if r["kind"] == "promotion" and r["reader"] == "nodef"}
    a = ["\\multicolumn{6}{l}{\\textit{A. Per attack family, per-item reader}} \\\\", "\\midrule",
         "& Attack family & ASR (\\%) & 95\\% CI & No def. & $n$ \\\\", "\\midrule"]
    for f, label in FAMILIES:
        r = fam.get(f) or {}
        q = nod.get(f) or {}
        a.append(L.row("R", f"{f}\\ {label}", *asr_cells(r), L.num(q.get("asr"), 1, 100.0), r.get("n", L.NOT_RUN)))
    dem = next((x for x in h1.get("pooled") or [] if x.get("kind") == "demotion" and x.get("reader") == "typed"), None)
    if dem:
        a.append(L.row("R", "A8\\ demotion (removed or halved)", *asr_cells(dem), L.NOT_RUN, dem["n"]))
    b = ["\\multicolumn{8}{l}{\\textit{B. Readers in the same slot, band and cap, A1--A7 pooled}} \\\\", "\\midrule",
         "& Reader & ASR (\\%) & 95\\% CI & $\\Delta$ (pp) & 95\\% CI & Holm $p$ & Loc. \\\\", "\\midrule"]
    pooled = {r["reader"]: r for r in h1["pooled"] if r["kind"] == "promotion"}
    sec = {c["reader"]: c for c in h1["secondary"]["contrasts"]}
    fam_p = {x["hypothesis"]: x for x in conf["family"]}
    c1 = h1["confirmatory"]
    for key, label, per_item in READERS:
        asr = asr_cells(pooled.get(key))
        lr = V.get(loc, "by_reader", key)
        locz = (L.pct(lr["rate"], 1) if lr and lr.get("status") == "ok" else L.NOT_RUN) if per_item else L.NA
        if key == "typed":
            b.append(L.row("R", label, *asr, "ref.", L.NA, L.NA, locz))
        elif key == c1.get("reader_b"):  # the H1 contrast (confirmatory)
            b.append(L.row("C", label, *asr, L.num(c1["estimate"], 1, 100.0), ci_col(*c1["ci"]),
                           p_fmt(fam_p["H1"]["p_holm"]), locz))
        else:
            s = sec.get(key) or {}
            ok = s.get("status") == "ok"
            b.append(L.row("R", label, *asr, L.num(s["estimate"], 1, 100.0) if ok else L.NOT_RUN,
                           ci_col(*s["ci"]) if ok else L.NOT_RUN, p_fmt(s.get("p_holm")) if ok else L.NOT_RUN, locz))
    rb = rta.get("budgets") or {}
    top = str(max(int(k) for k in rb)) if rb else None
    adaptive = (f" Adaptive attacker on NG ({top} queries per item, {rta.get('n_items')} items): surviving edit "
                f"{L.pct(rb[top]['survival_rate'], 0)}\\%, effect up one level "
                f"{L.pct(rb[top]['share_survivor_effect_up_1_level'], 0)}\\%.") if top else ""
    note = ("C: confirmatory (H1); R: registered. ASR: share of attacked stock-weeks whose name enters the signal "
            "book (A8: held name removed or halved), 95\\% interval clustered on the attack week; $^{*}$exact "
            "binomial. No def.: without guard questions. $\\Delta$: per-item reader minus this reader on matched stock-weeks (points); "
            f"Holm $p$ over the {h1['secondary']['m_registered']} secondary contrasts. Loc.: \\% of successes whose "
            "decision card ranks the planted item first." + adaptive)
    lines = ["% GENERATED by experiments/report/table3_planted.py -- do not edit by hand (CLAUDE.md rule 2).",
             "{\\scriptsize", "\\setlength{\\tabcolsep}{4pt}", *tab("clrlrr", a), "\\par\\smallskip",
             "{\\setlength{\\tabcolsep}{3pt}", *tab("clrlrlrr", b), "}", "\\par\\smallskip",
             "\\begin{minipage}{\\linewidth}\\scriptsize\\raggedright", note, "\\end{minipage}", "}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    docs = [V.require_registered(n) for n in ("h1", "confirmatory", "localization", "bound", "rta_ng")]
    p = L.write(L.TABLE_DIR, NAME, build(*docs))
    print(f"table3_planted -> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
