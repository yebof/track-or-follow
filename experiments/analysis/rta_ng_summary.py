#!/usr/bin/env python3
"""RT-A-NG, the adaptive attacker on the open typed model (prereg §10, decision P-4): the registered item-level
summary for the paper (Table 2's note), with the analysis provenance block.

    .venv/bin/python experiments/analysis/rta_ng_summary.py                      # registered: results/redteam/rta/ng
    .venv/bin/python experiments/analysis/rta_ng_summary.py --run-dir results/redteam/rta_ng/dry --window W_pre_slice

**Input.** The directory `experiments/redteam/rta_ng.py run` writes (results/redteam/rta/ng/): `asr_budget.json`
(`adaptive_ng.budget_summary` per budget) and `items.parquet` (the 30 C1 host items; their weeks). The run's spec hash
must equal the registered `NGSpec().sha256()` (quoted in the pre-registration); another hash is refused.

**Output** results/analysis/rta_ng.json: for each budget in {1, 4, 16, 64}, the share of items whose best candidate
survives N0 and the guards (the quarantine-bypass rate), the mean E[effect] shift of surviving candidates, and the
share of items with a surviving candidate one or more effect levels up, with the item count. Item level only: NG has
no Layer-B run, so there is no portfolio ASR and no transfer claim to emu; Table 2 reports it in a note, never in the
emu readers' ASR column. Descriptive (no test is registered).

**Lock.** The registered items are W_clean C1 units; the lock is checked on their weeks before the summary is read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "rta_ng"
DEFAULT_DIR = Path("results/redteam/rta/ng")
BUDGETS = ("1", "4", "16", "64")


def summarize(budget: dict) -> dict:
    out = {"spec_sha256": budget.get("spec_sha256"), "n_items": budget.get("n_items"), "budgets": {}}
    for b in BUDGETS:
        x = (budget.get("budgets") or {}).get(b) or {}
        out["budgets"][b] = {"n": x.get("n"), "survival_rate": x.get("survival_rate"),
                             "mean_delta_expected_survivors": x.get("mean_delta_expected_survivors"),
                             "mean_delta_expected_all": x.get("mean_delta_expected_all"),
                             "share_survivor_effect_up_1_level": x.get("share_survivor_effect_up_1_level")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default=None, help=f"the RT-A-NG output directory (default: {DEFAULT_DIR})")
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    d = Path(a.run_dir) if a.run_dir else root / DEFAULT_DIR
    if not (d / "asr_budget.json").exists():
        raise SystemExit(f"RT-A-NG: {C.rel(d, root)}/asr_budget.json missing; the registered run has not run")
    if (d / "items.parquet").exists():
        weeks = pd.to_datetime(pd.read_parquet(d / "items.parquet", columns=["t"])["t"]).unique()
        guard.check(weeks, "RT-A-NG items")  # before the summary is read
    else:
        guard.check_window(a.window, "RT-A-NG")
    from jevtrader.redteam.adaptive_ng import NGSpec

    budget = C.read_json(d / "asr_budget.json")
    want = NGSpec().sha256()
    if budget.get("spec_sha256") != want:
        raise SystemExit(f"RT-A-NG: spec {budget.get('spec_sha256')} is not the registered {want}")
    res = {**summarize(budget), "kind": budget.get("kind"),
           "scope": "item level on NG (typed-local): survival of N0 and the guards, E[effect] shift; no portfolio "
                    "replay, no transfer claim to emu"}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, res, script=__file__, window=a.window, guard=guard,
                   inputs={"asr_budget.json": d / "asr_budget.json",
                           "items.parquet": d / "items.parquet" if (d / "items.parquet").exists() else None},
                   argv=argv)
    b = res["budgets"]["64"]
    print(f"RT-A-NG {a.window}: {res['n_items']} items; survival at 64 queries {b['survival_rate']} -> "
          f"{C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
