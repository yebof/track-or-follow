"""Shared helpers of the exploratory analyses written AFTER the W_clean unlock (not pre-registered).

Every output of experiments/exploratory/ is labelled "exploratory (not pre-registered)": in its JSON `meta` (keys
`label`, `exploratory`), at the top of its payload, in every table or figure it writes, and in every macro it defines
(names prefixed \\ex..., comments carry the label). Outputs go to results/exploratory/ and paper/generated/
exploratory_numbers.tex, never to results/analysis/, so that experiments/report/make_numbers.py (which fills result
macros only from results/analysis/*.json) can never read them. Registered code is imported read-only; nothing here
edits a hashed file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "analysis"))

import common as C  # noqa: E402  (registered helpers, read-only)

LABEL = "exploratory (not pre-registered)"
OUT_DIR = ROOT / "results" / "exploratory"


def out_path(name: str) -> Path:
    return OUT_DIR / f"{name}.json"


def write(name: str, payload: dict, *, script: str | Path, window: str, guard, inputs=(), argv=None,
          why: str = "") -> Path:
    meta = {"label": LABEL, "exploratory": True,
            "why_exploratory": why or "analysis code written after the W_clean unlock (not hashed before the commit)"}
    return C.write_result(out_path(name), {"label": LABEL, **payload}, script=script, window=window, guard=guard,
                          inputs=inputs, argv=argv, extra_meta=meta)
