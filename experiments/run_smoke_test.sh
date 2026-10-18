#!/usr/bin/env bash
# Smoke test: checks that the pipeline runs end to end (~5 s). Not used in the report.
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py train --run-name smoke_test --embed-dim 50 --epochs 1 --max-window 5 --n-negatives 5 \
    --max-tokens 100000 --report-every 20000 --seed 42
python run.py evaluate --run-name smoke_test
