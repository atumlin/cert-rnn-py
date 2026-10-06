#!/usr/bin/env bash
# Step 6: tables and figures (add --hardware fair for the fair run)
# Resumable: rerun the same command after any interruption; finished work is skipped.
# Extra arguments are passed through (e.g. --bench grid9 --tools certrnn_plus crown, --quick).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m paper_bench.analyze "$@"
