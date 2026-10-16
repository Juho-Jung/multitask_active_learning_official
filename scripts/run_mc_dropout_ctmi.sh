#!/bin/bash
# Run the MC-Dropout CTMI experiment
# No conflict term; Hardness + MC-Dropout CTMI only

cd "$(dirname "$0")/.."

DEVICE=${1:-"cuda:0"}

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./runs/mtl_results/${TIMESTAMP}_mc_dropout_ctmi"
mkdir -p $SAVE_DIR

echo "=============================================="
echo "MC-Dropout CTMI Experiment"
echo "=============================================="
echo "Config: coco_mc_dropout_ctmi.json"
echo "Setup:"
echo "  - No conflict term (beta=0)"
echo "  - MC-Dropout CTMI (K=10 samples)"
echo "  - Hardness + CTMI"
echo "  - Diversity δ=0.5"
echo "Save dir: $SAVE_DIR"
echo "=============================================="

# Run experiment
python -m multitask_active_learning.runner \
    --config configs/coco/coco_mc_dropout_ctmi.json \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    2>&1 | tee $SAVE_DIR/run.log

echo "=============================================="
echo "Experiment completed!"
echo "Results: $SAVE_DIR"
echo "=============================================="
