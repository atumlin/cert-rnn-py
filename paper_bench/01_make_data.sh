#!/usr/bin/env bash
# Step 1: generate the grid benchmarks (Zone-A: --bench zoneA --data-dir DIR)
# Resumable: rerun the same command after any interruption; finished work is skipped.
# Extra arguments are passed through (e.g. --bench grid9 --tools certrnn_plus crown, --quick).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m paper_bench.make_data "$@"
