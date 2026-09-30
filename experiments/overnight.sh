#!/usr/bin/env bash
# Overnight battery: paper-critical first, each step independent so one
# failure doesn't kill the rest. Everything appends to results/*.jsonl
# (rerun-safe). Launch detached:
#   nohup setsid experiments/overnight.sh > experiments/results/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p experiments/results

step() {
    echo ""
    echo "=== [$(date '+%F %T')] $* ==="
}

step "E1 fast tools, all sizes (certrnn / interval / pgd)"
python -m experiments.suites.e1_verifiability --sizes S M L D \
    --tools certrnn-zono interval-ibp pgd || echo "STEP FAILED: e1-fast"

step "E1 lirpa-ibp, all sizes"
python -m experiments.suites.e1_verifiability --sizes S M L D \
    --tools lirpa-ibp --tag lirpaibp || echo "STEP FAILED: e1-lirpa-ibp"

step "E3 certified detection (synthetic center, both properties)"
python -m experiments.suites.e3_detection || echo "STEP FAILED: e3"

step "E4 full (gap, bisection depth, threat model)"
python -m experiments.suites.e4_tightness || echo "STEP FAILED: e4"

step "E2 scalability sweep H"
python -m experiments.suites.e2_scalability --sweep H || echo "STEP FAILED: e2-H"

step "E1 lirpa-backward (CROWN), cheap sizes first, 1h/case cap"
python -m experiments.suites.e1_verifiability --sizes S D M L \
    --tools lirpa-backward --tag crown || echo "STEP FAILED: e1-crown"

step "E2 scalability sweeps T, L (no CROWN: its budget story lives in the H sweep)"
python -m experiments.suites.e2_scalability --sweep T L \
    --tools certrnn-zono interval-ibp lirpa-ibp || echo "STEP FAILED: e2-TL"

step "E2 scalability sweep D (no CROWN)"
python -m experiments.suites.e2_scalability --sweep D \
    --tools certrnn-zono interval-ibp lirpa-ibp || echo "STEP FAILED: e2-D"

step "Summaries"
{
    echo "# Overnight results summary ($(date '+%F %T'))"
    echo
    echo "## E1"
    python -m experiments.tabulate experiments/results/e1.jsonl \
        experiments/results/e1_lirpaibp.jsonl \
        experiments/results/e1_crown.jsonl 2>/dev/null
    echo
    echo "## E3 (certified rate at eps=0.01)"
    python -m experiments.tabulate experiments/results/e3_normal.jsonl \
        experiments/results/e3_anom.jsonl --certified-at 0.01 2>/dev/null
    echo
    echo "## E2 scaling (H sweep)"
    python -m experiments.tabulate experiments/results/e2_sweepH.jsonl \
        --scaling 2>/dev/null
} > experiments/results/SUMMARY.md || echo "STEP FAILED: summary"

step "DONE"
