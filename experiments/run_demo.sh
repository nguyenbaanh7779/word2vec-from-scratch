#!/usr/bin/env bash
# Demo experiment: 1M tokens x 3 epochs (~4 min on an Apple M5). Used as the smaller-corpus comparison in the report.
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py train --run-name demo --embed-dim 100 --epochs 3 --max-window 5 --n-negatives 5 \
    --max-tokens 1000000 --report-every 100000 --seed 42
python run.py evaluate --run-name demo
