#!/bin/bash
# COCO 2-Stage Selection Optimized Experiment
# 목표: 베이스라인 (diversity: 0.7728)을 초과하는 성능 달성

set -e

# Configuration
CONFIG_FILE="configs/coco_2stage_selection_optimized.json"
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
cd /opt/multitask_active_learning

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
