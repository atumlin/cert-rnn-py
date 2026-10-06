#!/usr/bin/env bash
# Step 5: the fair run -- every tool on one CPU process with the same budgets
# (fixed-eps checks, then certified radius). Run after the recommended run.
# Resumable: rerun the same command after any interruption.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m paper_bench.check --hardware fair "$@"
python -m paper_bench.radius --hardware fair "$@"
