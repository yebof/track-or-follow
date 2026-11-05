#!/usr/bin/env python3
"""The BOUND claim (prereg §3.1 "Bound (not a hypothesis; must hold)", §4.2 claim register row BOUND).

    .venv/bin/python experiments/analysis/bound_claim.py                   # registered: the primary cell on W_clean
    .venv/bin/python experiments/analysis/bound_claim.py --window W_pre_slice --gen-id pre_blinded_emu_wpre_s30

**Inputs.** (1) The primary run's `bound_check.json` (experiments/run_layer_b.py --bound-check; eval.bound.bound_check):
the BG-5 worst-case-reader check over every learned week and name (entry and protection checks; the trusted side
must read T-channel features only) and the BG-6 constructed witnesses. (2) The property tests of Propositions P0-P5:
tests/test_band_guard_bound.py (P0-P4 under randomized attacks, BG-4/BG-5/BG-6) and tests/test_bound_history.py (P5,
poisoned history), run here by pytest in a subprocess; the pass and fail counts and the SHA-256 of both files are
written (`--skip-tests` records them as not run, development only).

**Claim.** `claim.bound` is True iff the worst-case reader passed (0 entry violations, 0 protection violations,
trusted side all T) AND both property-test files passed with no failure. Any violation retracts the claim (§4.2)
and the output lists it. `confirmatory.py` copies the claim into `claims.BOUND` (outside the Holm family), and
make_numbers states "the influence bound capped the damage for every reader" only then.

**Lock.** The bound check replays learned coefficients of the window, so the run's window is lock-checked
(common.read_eval_run) before bound_check.json is read. Output: results/analysis/bound.json.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "bound"
DEFAULT_GEN = "main_emu_wclean_u100"
PROPERTY_TESTS = ("tests/test_band_guard_bound.py", "tests/test_bound_history.py")


def run_property_tests(root: Path) -> dict:
    """pytest on the property-test files; counts parsed from the summary line."""
    files = [str(Path(C.ROOT) / f) for f in PROPERTY_TESTS]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
                          cwd=str(C.ROOT), capture_output=True, text=True, check=False)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", tail[0])}
    return {"status": "run", "returncode": proc.returncode, "summary": tail[0], "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0),
            "skipped": counts.get("skipped", 0),
            "files": {f: C.sha256_file(Path(C.ROOT) / f) for f in PROPERTY_TESTS}}


def decide(bc: dict | None, tests: dict) -> dict:
    wc = (bc or {}).get("worst_case_reader") or {}
    bg5 = {"status": "ok" if wc else "not run", "n_weeks": wc.get("n_weeks"),
           "n_entry_checks": wc.get("n_entry_checks"), "n_protection_checks": wc.get("n_protection_checks"),
           "n_cases": (wc.get("n_entry_checks") or 0) + (wc.get("n_protection_checks") or 0) if wc else None,
           "entry_violations": wc.get("entry_violations", []), "protection_violations":
           wc.get("protection_violations", []), "trusted_all_T": wc.get("trusted_all_T"),
           "n_violations": len(wc.get("entry_violations", [])) + len(wc.get("protection_violations", []))
           if wc else None, "passed": bool(wc.get("passed")) if wc else False}
    tests_ok = tests.get("status") == "run" and tests.get("returncode") == 0 and tests.get("failed", 1) == 0 \
        and tests.get("passed", 0) > 0
    return {"bg5": bg5, "witnesses_bg6": (bc or {}).get("constructed_witnesses"), "property_tests": tests,
            "claim": {"bound": bool(bg5["passed"] and tests_ok),
                      "rule": "BG-5 worst-case reader passed (0 violations, trusted side all T) and the P0-P5 "
                              "property tests passed"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--primary", default=None, help="Layer-B run dir of the primary cell (default: found by name)")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the property tests (development only)")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    guard.check_window(a.window, "BOUND")
    pdir = Path(a.primary) if a.primary else C.find_eval_dir(root, a.window, "primary", a.gen_id)
    run = C.read_eval_run(pdir, guard, "BOUND primary")  # the lock check on the run's window and dates
    f = run.file("bound_check.json")
    bc = C.read_json(f) if f.exists() else None
    tests = {"status": "not run (--skip-tests)"} if a.skip_tests else run_property_tests(root)
    res = decide(bc, tests)
    res["primary_run"] = {"dir": C.rel(pdir, root), "bound_check_json": f.exists(),
                          "cfg_sha": run.metrics.get("cfg_sha"), "inputs_sha": run.metrics.get("inputs_sha")}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, res, script=__file__, window=a.window, guard=guard,
                   inputs={"bound_check.json": f if f.exists() else None}, argv=argv)
    b = res["bg5"]
    print(f"BOUND {a.window}: BG-5 {b['status']} ({b['n_cases']} cases, {b['n_violations']} violations); property "
          f"tests {tests.get('summary', tests['status'])}; claim {res['claim']['bound']} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
