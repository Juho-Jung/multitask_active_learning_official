#!/bin/bash
# COCO 2-Stage Selection Optimized Experiment

set -e
cd "$(dirname "$0")/.."

# Configuration
CONFIG_FILE="configs/coco/coco_2stage_selection_optimized.json"
SAVE_DIR="./runs/mtl_results"
DEVICE="cuda:0"

# Create timestamped save directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${SAVE_DIR}/${TIMESTAMP}_coco_2stage_optimized"

# Create directory if it doesn't exist
mkdir -p "${EXPERIMENT_DIR}"

echo "=================================================================================="
echo "🚀 COCO 2-Stage Selection Optimized Experiment"
echo "=================================================================================="
echo "Config: ${CONFIG_FILE}"
echo "Save Dir: ${EXPERIMENT_DIR}"
echo "Device: ${DEVICE}"
echo "=================================================================================="
echo ""

# Run experiment
python -m multitask_active_learning.runner \
    --config "${CONFIG_FILE}" \
    --save_dir "${EXPERIMENT_DIR}" \
    --device "${DEVICE}" \
    2>&1 | tee "${EXPERIMENT_DIR}/run.log"

echo ""
echo "=================================================================================="
echo "✅ Experiment completed!"
echo "Results saved to: ${EXPERIMENT_DIR}"
echo "=================================================================================="
