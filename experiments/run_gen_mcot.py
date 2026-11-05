"""MCOT (B15, Mono-CoT): S30 x the last 60 W_clean decisions, the same 1,800 stock-weeks and inputs as TAL.

A wrapper around the gens operator's runner (`experiments/gens/run_gen.py --reader mcot`, unchanged): frozen bank v1.0
`mono-cot-v1`, pinned gpt-4.1-mini-2025-04-14, `gen-parse-v1`, one call per stock-week, weeks newest first, resumable
per week. Only two things differ from gens_report §3, both decided by the orchestrator on 2026-09-28:
- the MCOT sub-cap is $4.50 (configs/backends.yaml; P-2 keeps the baseline at S30 x 60);
- P-2 drops the $75 reserve. The task's stop rule replaces it: the list stops if the ledger would exceed $225. The
  runner's `RESERVE` (the room it keeps below the $250 budget) is therefore set to 250 - 225 = $25, so its preflight
  aborts, and its process guard stops, before the ledger could pass $225. The sub-cap and the $245 hard cap apply as
  before (SpendLedger on every call).

    .venv/bin/python experiments/run_gen_mcot.py dry        # $0: fake client + projection (scratch)
    .venv/bin/python experiments/run_gen_mcot.py run        # preflight, then paid (tag MCOT/mono-cot)
    .venv/bin/python experiments/run_gen_mcot.py validate   # experiments/gens/validate_gen.py

Output: results/judgments/mcot_mono-cot_wclean-last60_s30/ (answers, requests, outputs, MANIFEST.json, shards/,
projection.json, validation.json). Layer-A outputs only; nothing reads a W_clean outcome.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENS = ROOT / "experiments" / "gens"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(GENS))

GEN_ID = "mcot_mono-cot_wclean-last60_s30"
OUT = ROOT / "results" / "judgments" / GEN_ID
SCRATCH = Path("/tmp/jevtrader-scratch/paid")
STOP_LEDGER = 225.0
PY = str(ROOT / ".venv" / "bin" / "python")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "dry":
        d = SCRATCH / f"mcot_dry_{int(time.time())}"
        r = subprocess.run([PY, str(GENS / "run_gen.py"), "--reader", "mcot", "--dry", "--out", str(d)])
        if r.returncode:
            return r.returncode
        return subprocess.run([PY, str(GENS / "projection.py"), "--dry-dir", f"mcot={d}",
                               "--out", str(OUT / "projection.json")]).returncode
    if cmd == "run":
        import run_gen as RG

        RG.RESERVE = RG.USER_BUDGET - STOP_LEDGER  # P-2: the $225 stop threshold replaces the $75 reserve
        sys.argv = [str(GENS / "run_gen.py"), "--reader", "mcot", "--out", str(OUT),
                    "--projection", str(OUT / "projection.json"), *sys.argv[2:]]
        return RG.main()
    if cmd == "validate":
        return subprocess.run([PY, str(GENS / "validate_gen.py"), "--reader", "mcot", "--gen", str(OUT),
                               "--ledger", str(ROOT / "data/cache/spend.sqlite"), "--out", str(OUT / "validation.json")]).returncode
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
