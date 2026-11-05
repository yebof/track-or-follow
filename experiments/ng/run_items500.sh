#!/usr/bin/env bash
# NG item-level sample, the long job (experiments/ng/items500.py; docs/impl/ng_report.md). Free: local CPU, $0 API.
# Resumable: rerun this script after any stop; finished items are skipped (checkpoint.jsonl).
#
#   nohup experiments/ng/run_items500.sh > results/ng/logs/items500_run.log 2>&1 &
#
# Steps: sample (refuses to change an existing sample) -> prepare (skips verified weeks) -> run (one item per
# checkpoint line) -> finalize (parquet tables + MANIFEST.json) -> verify (all finished items).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
S=experiments/ng/items500.py
echo "$(date -u +%FT%TZ) start pid $$"
echo $$ > results/ng/logs/items500_run.pid
$PY $S sample
nice -n 10 $PY $S prepare
nice -n 10 $PY $S run
$PY $S finalize
$PY $S verify --first 0 || true
echo "$(date -u +%FT%TZ) end"
