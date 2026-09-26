#!/bin/bash
# Rebuttal Ablation Experiments
#
# Runs α,γ sensitivity + component-wise ablation for the rebuttal.
# Each variant runs ODS (2-stage) on COCO with different configurations.
#
# Usage:
#   bash scripts/run_rebuttal_ablations.sh            # Run all
#   bash scripts/run_rebuttal_ablations.sh sensitivity # α,γ sensitivity only
#   bash scripts/run_rebuttal_ablations.sh component   # Component ablation only
#   bash scripts/run_rebuttal_ablations.sh dry-run     # Print configs only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Rebuttal Ablation Experiments"
echo " Project: $PROJECT_DIR"
echo "============================================"

EXTRA_ARGS=""
if [ "$1" = "sensitivity" ]; then
    EXTRA_ARGS="--only sensitivity"
    echo " Mode: α, γ Sensitivity only"
elif [ "$1" = "component" ]; then
    EXTRA_ARGS="--only component"
    echo " Mode: Component-wise Ablation only"
elif [ "$1" = "dry-run" ]; then
    EXTRA_ARGS="--dry-run"
    echo " Mode: Dry run (no execution)"
fi

echo ""
python "$SCRIPT_DIR/run_rebuttal_ablations.py" $EXTRA_ARGS

echo ""
echo "============================================"
echo " Analyzing results..."
echo "============================================"

if [ "$1" != "dry-run" ]; then
    python "$SCRIPT_DIR/analyze_rebuttal_ablations.py"
fi

echo ""
echo " Done! Check results/rebuttal_ablations/ and rebuttal/ablation_tables.tex"
