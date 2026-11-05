#!/usr/bin/env python3
"""Budget snapshot for the pre-registration's stop and budget rule (prereg §13): the spend ledger (read-only) against
the hard cap and the per-run sub-caps of configs/backends.yaml (development tool; no model call, no outcome).

    .venv/bin/python experiments/analysis/budget_snapshot.py     # -> results/analysis/budget_snapshot.json

For every run key (the first segment of a ledger tag): spend, sub-cap (0 = disabled) and the remaining headroom; the
ledger total, the $250 user budget and the hard cap; and the worst case the rule bounds: the ledger if every run with
a positive sub-cap spent all of its remaining headroom.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path("results/analysis/budget_snapshot.json")
USER_BUDGET = 250.0


def snapshot(root: Path = C.ROOT) -> dict:
    cfg = yaml.safe_load((root / "configs/backends.yaml").read_text())
    led = cfg["ledger"]
    caps = {k: float(v) for k, v in (led.get("sub_caps_usd") or {}).items()}
    db = root / led.get("path", "data/cache/spend.sqlite")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT tag, COUNT(*), COALESCE(SUM(usd), 0), MAX(ts) FROM spend GROUP BY tag").fetchall()
    n, total, last = con.execute("SELECT COUNT(*), COALESCE(SUM(usd), 0), MAX(ts) FROM spend").fetchone()
    con.close()
    spend: dict[str, float] = {}
    for tag, _, usd, _ in rows:
        spend[str(tag).split("/", 1)[0]] = spend.get(str(tag).split("/", 1)[0], 0.0) + float(usd)
    runs = {}
    for k in sorted(set(caps) | set(spend)):
        cap = caps.get(k)
        s = spend.get(k, 0.0)
        runs[k] = {"spent_usd": round(s, 6), "sub_cap_usd": cap,
                   "headroom_usd": None if not cap else round(max(cap - s, 0.0), 6),
                   "status": "disabled (sub-cap 0)" if cap == 0 else ("no sub-cap entry" if cap is None else
                                                                         "enabled")}
    headroom = sum(v["headroom_usd"] or 0.0 for v in runs.values())
    hard = float(led["hard_cap_usd"])
    return {"ledger": {"rows": int(n), "total_usd": round(float(total), 6), "last_ts": last},
            "hard_cap_usd": hard, "user_budget_usd": USER_BUDGET,
            "remaining_of_user_budget_usd": round(USER_BUDGET - float(total), 6),
            "remaining_below_hard_cap_usd": round(hard - float(total), 6),
            "runs": runs, "sum_enabled_headroom_usd": round(headroom, 6),
            "worst_case_ledger_if_every_enabled_run_spends_its_headroom_usd": round(float(total) + headroom, 6),
            "worst_case_within_hard_cap": float(total) + headroom <= hard}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(C.ROOT / OUT))
    a = ap.parse_args(argv)
    res = snapshot()
    out = Path(a.out)
    C.write_result(out, res, script=__file__, window="budget", guard=C.Guard(C.ROOT),
                   inputs={"backends.yaml": C.ROOT / "configs/backends.yaml"}, argv=argv)
    print(f"ledger ${res['ledger']['total_usd']:.4f}; worst case "
          f"${res['worst_case_ledger_if_every_enabled_run_spends_its_headroom_usd']:.2f} vs hard cap "
          f"${res['hard_cap_usd']:.0f} -> {C.rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
