#!/usr/bin/env bash
# Reproduce the benchmark results end to end (artifact evaluation).
#   paper_bench/reproduce.sh --quick     ~1 h smoke version (tiny data, 2 windows, 1 eps)
#   paper_bench/reproduce.sh --full      everything in config.yaml (days; see README.md)
# Every step is resumable: if anything stops, run the same command again.
# Zone-A (Sandia only) is not part of this script: see zoneA steps in README.md.
set -euo pipefail
MODE="${1:---quick}"
D="$(dirname "${BASH_SOURCE[0]}")"
Q=(); [[ "$MODE" == "--quick" ]] && Q=(--quick)
"$D/01_make_data.sh" ${Q[@]+"${Q[@]}"}
"$D/02_train.sh" ${Q[@]+"${Q[@]}"}
"$D/03_check.sh" ${Q[@]+"${Q[@]}"}
"$D/04_radius.sh" ${Q[@]+"${Q[@]}"}
"$D/06_analyze.sh" ${Q[@]+"${Q[@]}"}
if [[ "$MODE" == "--full" ]]; then
  "$D/05_fair.sh"
  "$D/06_analyze.sh" --hardware fair
fi
source "$D/_env.sh" && python -m paper_bench.status ${Q[@]+"${Q[@]}"}
