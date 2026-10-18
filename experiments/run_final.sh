#!/usr/bin/env bash
# Final experiment: 5M tokens x 3 epochs (~25 min on an Apple M5). Main results of the report and slides.
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py train --run-name final --embed-dim 100 --epochs 3 --max-window 5 --n-negatives 5 \
    --max-tokens 5000000 --report-every 500000 --seed 42
python run.py evaluate --run-name final
