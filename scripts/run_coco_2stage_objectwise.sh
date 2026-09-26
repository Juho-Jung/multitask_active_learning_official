#!/bin/bash
# COCO 2-Stage Selection with Object-wise Hardness

set -euo pipefail

CONFIG="configs/coco_2stage_selection_objectwise.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="coco_2stage_objectwise"
SAVE_DIR="./runs/mtl_results/${TIMESTAMP}_${EXPERIMENT_NAME}"
DEVICE=${1:-"cuda:0"}

echo "=================================================================================="
echo "🚀 COCO 2-Stage Selection with Object-wise Hardness"
echo "=================================================================================="
echo "Config: $CONFIG"
echo "Save Dir: $SAVE_DIR"
echo "Device: $DEVICE"
echo "=================================================================================="
echo ""
echo "📋 Key Improvements:"
echo "  - Detection: Top-K box entropy sum (instead of mean)"
echo "  - Segmentation: 90th percentile entropy (instead of mean)"
echo "  - Reduced CTMI weight: γ=0.4 (was 1.0)"
echo "  - Increased pool factor: m=12 (was 5-8)"
echo "  - Increased hardness weight: α=2.0 (was 1.0)"
echo "=================================================================================="
echo ""

# Create save directory
mkdir -p "${SAVE_DIR}"

# Run experiment
cd /opt/multitask_active_learning

SAVE_MODEL_PATH="./runs/checkpoints/${TIMESTAMP}_${EXPERIMENT_NAME}"

# Run the experiment
python3 -m multitask_active_learning.runner \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    --save_model_path "$SAVE_MODEL_PATH" \
    --device "$DEVICE" \
    2>&1 | tee "${SAVE_DIR}/run.log"

exit_code=${PIPESTATUS[0]}

# Save config file to method-specific directory (even if experiment failed)
# The runner creates subdirectories like ${SAVE_DIR}/ours-2stage/ and saves config.json
# But we also save the original config file with its original name for reference
METHOD_DIR="${SAVE_DIR}/ours-2stage"
if [ -d "$METHOD_DIR" ]; then
    CONFIG_BASENAME=$(basename "$CONFIG")
    cp "$CONFIG" "${METHOD_DIR}/${CONFIG_BASENAME}"
    echo "📋 Original config file saved to: ${METHOD_DIR}/${CONFIG_BASENAME}"
fi

# Also save to main save_dir
CONFIG_BASENAME=$(basename "$CONFIG")
cp "$CONFIG" "${SAVE_DIR}/${CONFIG_BASENAME}"
echo "📋 Original config file saved to: ${SAVE_DIR}/${CONFIG_BASENAME}"

if [ "$exit_code" -ne 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "❌ Experiment failed (exit code: $exit_code). See log: ${SAVE_DIR}/run.log"
    echo "=================================================================================="
    exit "$exit_code"
fi

echo ""
echo "=================================================================================="
echo "✅ Experiment completed! Results saved to: $SAVE_DIR"
echo "=================================================================================="
