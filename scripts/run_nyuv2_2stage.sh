#!/bin/bash
# Run the 2-stage selection (ODS) experiment on NYUv2

CONFIG_FILE="configs/nyuv2/nyuv2_2stage_selection.json"
SAVE_DIR="./runs/mtl_results/$(date +%Y%m%d_%H%M%S)_nyuv2_2stage_selection"
DEVICE=${1:-"cuda:0"}

echo "=========================================="
echo "NYUv2 2-Stage Selection Experiment"
echo "=========================================="
echo "Config: nyuv2_2stage_selection.json"
echo ""
echo "Setup:"
echo "  - Stage 1: Hardness + CTMI (Informativeness)"
echo "  - Stage 2: K-Center Greedy (Diversity)"
echo "  - No conflict term (beta=0)"
echo "  - pool_factor=5 (candidate pool = 5x batch)"
echo ""
echo "Save dir: $SAVE_DIR"
echo "Device: $DEVICE"
echo "=========================================="

python -m multitask_active_learning.runner \
    --config $CONFIG_FILE \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "Experiment completed!"
echo "Results: $SAVE_DIR"
echo "=========================================="
