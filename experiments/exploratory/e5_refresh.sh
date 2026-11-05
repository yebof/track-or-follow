#!/usr/bin/env bash
# EXPLORATORY (not pre-registered): rebuild every E5 output and the floats that read it, in dependency order ($0).
# Run after the v1.1 full-frame coding (`e5_construct_checks.py code-frame`) completes, or after any E5 input changes:
#   experiments/exploratory/e5_refresh.sh
# With --wait-pid PID it first waits for that process (the background coder) to exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == "--wait-pid" ]]; then
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
fi
PY=.venv/bin/python
$PY experiments/exploratory/e5_construct_checks.py run
$PY experiments/exploratory/e5_box1_examples.py render
$PY experiments/exploratory/e5_make_numbers.py
$PY experiments/exploratory/e6_publication_timing.py run   # reads e5_construct_checks.json (residual approximation)
$PY experiments/exploratory/r1_make_numbers.py
$PY experiments/exploratory/r2_make_numbers.py   # round-2 macros (E8, E2 fresh items, analysis count)
$PY experiments/exploratory/e5_table_concepts.py
$PY experiments/report/fig2_validity.py
echo "$(date -u +%FT%TZ) e5_refresh done"
