"""Shared helpers of the validity-first floats (outline_v2 §3, §4): the reader rows, their marks, and the Jev slot.

Readers behind the one typed-judgment interface (positioning_v3 §1.2): emu (the confirmatory reader: a pinned LLM's
option probabilities), VERB (the same LLM asked for verbalized probabilities; a contrast) and NG (an open
non-generative typed model; descriptive). **Jev slot (floats):** a Jev row of Table 1 / Fig. 2 exists only if a
registered-window, lock-open `results/analysis/h2.json` carries `per_implementation.<Jev>` with status "ok" (key
matched case-insensitively); none does (the Jev panel is descriptive, not an H2 implementation row), so those floats
print no Jev element. **Jev panel (macros, Table A1):** `jev_rows()` returns results/analysis/jev_rows.json
(experiments/jev_panel/jev_rows.py) only if it carries the descriptive label, was written with the lock open and every
panel generation it read passed validation; make_numbers_extra.py's Jev macros key on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402  (read-only)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results" / "analysis"
EXPLORATORY = ROOT / "results" / "exploratory"
LABEL_X = "exploratory (not pre-registered)"
C = style.COLORS

# reader -> (display label, colour, marker); estimate filled, placebo open (outline_v2 §4). Round-2 figure review J-2:
# every series has its own shape: Jev a hexagon, the reference readers a thin diamond (FinBERT) and a filled x
# (dictionary), so that no two series share a shape and a similar grey.
READERS = {"emu": ("emu", C["indigo"], "o"), "VERB": ("VERB", C["wine"], "^"), "NG": ("NG", C["teal"], "s"),
           "Jev": ("Jev", C["olive"], "h")}
DICT_COLOR, DICT_MARKER = C["grey"], "X"
FINBERT_MARKER = "d"
PANEL_X_BG = "#F7F7F7"  # exploratory panels (outline_v2 §4 status encoding)

SEC_CONCEPTS = (("earnings_results_vs_2.02", "Earnings results", "2.02"),
                ("deal_or_investment_vs_1.01/2.01", "Deal or investment", "1.01/2.01"),
                ("management_or_labor_vs_5.02", "Management or labor", "5.02"))

EVENT_TYPE_LABEL = {
    "earnings_results": "Earnings results", "guidance_change": "Guidance change",
    "analyst_action": "Analyst action", "deal_or_investment": "Deal or investment",
    "product_or_contract": "Product or contract", "capital_return": "Capital return",
    "financing_or_credit": "Financing or credit", "legal_or_regulatory": "Legal or regulatory",
    "management_or_labor": "Management or labor", "operations_disruption": "Operations disruption",
    "industry_or_macro_readthrough": "Industry or macro read-through", "other": "Other development"}
# 8-K concept that an event type maps to (h2_validity.SEC_CONCEPTS), for Table 2's registered column
TYPE_TO_SEC = {"earnings_results": "earnings_results_vs_2.02", "deal_or_investment": "deal_or_investment_vs_1.01/2.01",
               "management_or_labor": "management_or_labor_vs_5.02"}


def registered(name: str) -> dict | None:
    """results/analysis/<name>.json only if written for the registered window with the lock open (the
    make_numbers.registered_result guard)."""
    p = ANALYSIS / f"{name}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    m = d.get("meta", {})
    return d if m.get("registered_window") and (m.get("lock") or {}).get("open") else None


def require_registered(name: str) -> dict:
    d = registered(name)
    if d is None:
        raise SystemExit(f"refused: results/analysis/{name}.json is missing or not a registered, lock-open output")
    return d


def exploratory(name: str) -> dict | None:
    """results/exploratory/<name>.json only if its meta says exploratory, with the label, written with the lock open
    (the make_exploratory.load guard)."""
    p = EXPLORATORY / f"{name}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    m = d.get("meta", {})
    if not (m.get("exploratory") is True and m.get("label") == LABEL_X and (m.get("lock") or {}).get("open")):
        raise SystemExit(f"{p}: not an exploratory output written with the lock open")
    return d


def jev_key(h2: dict | None) -> str | None:
    """The per_implementation key of a Jev row with status ok, or None (the Jev slot stays empty)."""
    impl = (h2 or {}).get("per_implementation") or {}
    for k, v in impl.items():
        if k.lower() == "jev" and isinstance(v, dict) and v.get("status") == "ok":
            return k
    return None


JEV_ROWS = ANALYSIS / "jev_rows.json"
JEV_ROWS_LABEL = "descriptive (Jev panel; not part of H1-H3)"


def jev_rows() -> dict | None:
    """results/analysis/jev_rows.json if it is the descriptive Jev-panel output (label, lock open, every validated
    generation passed); None otherwise (the Jev macros then print nothing)."""
    if not JEV_ROWS.is_file():
        return None
    d = json.loads(JEV_ROWS.read_text())
    m = d.get("meta", {})
    gens = m.get("generations") or {}
    ok = (m.get("label") == JEV_ROWS_LABEL and m.get("descriptive") is True and (m.get("lock") or {}).get("open")
          and {"forward", "aligned2000", "h2frame", "probes"} <= set(gens)
          and all(g.get("validation_pass") is True for k, g in gens.items() if k != "canary"))
    if not ok:
        raise SystemExit(f"{JEV_ROWS}: not the descriptive Jev-panel output (label, lock or validation)")
    return d


def get(d, *path, default=None):
    for k in path:
        if isinstance(d, list) and isinstance(k, int):
            d = d[k] if -len(d) <= k < len(d) else None
        elif isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
        if d is None:
            return default
    return d


def tabular(colspec: str, rows: list[str], note: str | None = None, size: str = "\\footnotesize",
            width: str | None = None, sep: str = "3pt", source: str = "experiments/report/") -> str:
    """A booktabs tabular (outline_v1 §4 "Tables") whose note is set flush left at the table's width. With `width`
    (for example "\\linewidth"), a tabular* stretched to that width with the column space distributed evenly.
    `rows` holds everything between \\toprule and \\bottomrule, header lines included."""
    if width:
        import _xtab as XT  # tabularx: alternating row colours without gaps (user, 2026-10-22)
        head, tail = XT.begin(width, colspec), XT.END
    else:
        head, tail = f"\\begin{{tabular}}{{{colspec}}}", "\\end{tabular}"
    lines = [f"% GENERATED by {source} -- do not edit by hand (CLAUDE.md rule 2).", "{" + size,
             f"\\setlength{{\\tabcolsep}}{{{sep}}}", head, "\\toprule", *rows, "\\bottomrule", tail]
    if note:
        lines += ["\\par\\smallskip", "\\begin{minipage}{" + (width or "\\linewidth") + "}\\scriptsize\\raggedright",
                  note, "\\end{minipage}"]
    lines.append("}")
    return "\n".join(lines) + "\n"
