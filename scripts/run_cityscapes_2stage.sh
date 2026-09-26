#!/bin/bash
# Run Cityscapes 2-Stage Selection Experiment
# Tasks: Semantic Segmentation (19 classes), Depth Estimation, Instance Segmentation

set -e

# Configuration
CONFIG_FILE="configs/cityscapes_2stage_selection.json"
DEVICE=${1:-"cuda:0"}
RESULTS_DIR="./runs/mtl_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="cityscapes_2stage_selection"

echo "=============================================="
echo "Cityscapes 2-Stage Selection Experiment"
echo "=============================================="
echo "Config: ${CONFIG_FILE}"
echo "Device: ${DEVICE}"
echo "Results: ${RESULTS_DIR}/${TIMESTAMP}_${EXPERIMENT_NAME}"
echo "=============================================="

# Create results directory
mkdir -p "${RESULTS_DIR}/${TIMESTAMP}_${EXPERIMENT_NAME}"

# Run experiment
cd /opt/multitask_active_learning

python -u -m multitask_active_learning.runner \
    --config "${CONFIG_FILE}" \
    --device "${DEVICE}" \
    --save_dir "${RESULTS_DIR}/${TIMESTAMP}_${EXPERIMENT_NAME}" \
    2>&1 | tee "${RESULTS_DIR}/${TIMESTAMP}_${EXPERIMENT_NAME}/experiment.log"

echo "=============================================="
echo "Experiment completed!"
echo "Results saved to: ${RESULTS_DIR}/${TIMESTAMP}_${EXPERIMENT_NAME}"
echo "=============================================="
