#!/bin/bash
# Rebuttal: Pretrained ResNet-50 Backbone Comparison
#
# Compares ODS vs. baselines with ImageNet-pretrained ResNet-50 encoder.
# Addresses Reviewer 3's concern about pretrained model usage.
#
# Usage:
#   bash scripts/run_pretrained_comparison.sh                    # Run all
#   bash scripts/run_pretrained_comparison.sh dry-run            # Print configs only
#   bash scripts/run_pretrained_comparison.sh "ours-2stage random" # Specific methods

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Rebuttal: Pretrained ResNet-50 Comparison"
echo "============================================"

EXTRA_ARGS=""
if [ "$1" = "dry-run" ]; then
    EXTRA_ARGS="--dry-run"
    echo " Mode: Dry run (no execution)"
elif [ -n "$1" ]; then
    EXTRA_ARGS="--methods $1"
    echo " Mode: Selected methods — $1"
fi

echo ""
python "$SCRIPT_DIR/run_pretrained_comparison.py" $EXTRA_ARGS

echo ""
echo "============================================"
echo " Analyzing results..."
echo "============================================"

if [ "$1" != "dry-run" ]; then
    python "$SCRIPT_DIR/analyze_pretrained_comparison.py"
fi

echo ""
echo " Done! Check results/rebuttal_pretrained/ and rebuttal/pretrained_comparison_table.tex"
