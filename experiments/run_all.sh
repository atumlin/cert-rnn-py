#!/usr/bin/env bash
# Full experiment battery for the paper. Order: cheap -> expensive.
# Every suite is resumable at the JSONL level; rerun-safe (appends).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m experiments.suites.e4_tightness --quick          # ~minutes: sanity + gap preview
python -m experiments.suites.e1_verifiability --sizes S M D
python -m experiments.suites.e3_detection
python -m experiments.suites.e2_scalability --sweep H T L
python -m experiments.suites.e1_verifiability --sizes L    # slowest last
python -m experiments.suites.e2_scalability --sweep D
python -m experiments.suites.e4_tightness                  # full ablations

python -m experiments.tabulate experiments/results/e1.jsonl
