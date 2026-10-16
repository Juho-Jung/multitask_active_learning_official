#!/bin/bash
# Run the 2-stage selection (ODS) experiment on COCO.
#
# - Avoids diversity dilution: diversity is applied as a separate stage (K-Center Greedy)
# - Gradient conflict is excluded from the acquisition score (beta=0)
# - Diversity is computed only on the Stage-1 candidate pool (pool_factor=5)

cd "$(dirname "$0")/.."

DEVICE=${1:-"cuda:0"}

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./runs/mtl_results/${TIMESTAMP}_2stage_selection"
mkdir -p $SAVE_DIR

echo "=============================================="
echo "2-Stage Selection Experiment"
echo "=============================================="
echo "Config: coco_2stage_selection.json"
echo ""
echo "Setup:"
echo "  - Stage 1: Hardness + CTMI (Informativeness)"
echo "  - Stage 2: K-Center Greedy (Diversity)"
echo "  - No conflict term (beta=0)"
echo "  - pool_factor=5 (candidate pool = 5x batch)"
echo ""
echo "Save dir: $SAVE_DIR"
echo "=============================================="

# Run experiment
python -m multitask_active_learning.runner \
    --config configs/coco/coco_2stage_selection.json \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    2>&1 | tee $SAVE_DIR/run.log

echo "=============================================="
echo "Experiment completed!"
echo "Results: $SAVE_DIR"
echo "=============================================="
