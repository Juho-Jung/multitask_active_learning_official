#!/bin/bash
# NYUv2 Baseline 실험 실행 스크립트

BASE_CONFIG="configs/nyuv2_2stage_selection.json"
BASE_SAVE_DIR="./runs/mtl_results"
EXPERIMENT_NAME="nyuv2_baselines_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${BASE_SAVE_DIR}/${EXPERIMENT_NAME}"
DEVICE="cuda:3"
TMP_CONFIG_DIR="/tmp/nyuv2_baseline_configs"

# Baseline methods
BASELINE_METHODS=("random" "uncertainty" "diversity" "coreset" "badge" "batchbald" "clustering")

echo "=========================================="
echo "NYUv2 Baseline 실험 시작"
echo "=========================================="
echo "Base config: $BASE_CONFIG"
echo "Baseline methods: ${BASELINE_METHODS[@]}"
echo "Save dir: $SAVE_DIR"
echo "Device: $DEVICE"
echo "=========================================="

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$TMP_CONFIG_DIR"

# Run each baseline method
for method in "${BASELINE_METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running baseline: $method"
    echo "=========================================="
    
    # Create temporary config file for this method
    TMP_CONFIG="${TMP_CONFIG_DIR}/nyuv2_${method}.json"
    python3 << EOF
import json
import sys

# Load base config
with open('$BASE_CONFIG', 'r') as f:
    config = json.load(f)

# Update methods to only include current baseline
config['methods'] = ['$method']

# Save temporary config
with open('$TMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Created config: $TMP_CONFIG")
print(f"Methods: {config['methods']}")
print(f"Tasks: {config.get('tasks', [])}")
print(f"Dataset: {config.get('dataset', 'unknown')}")
EOF
    
    # Run experiment
    python -m multitask_active_learning.runner \
        --config "$TMP_CONFIG" \
        --save_dir "$SAVE_DIR" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✅ $method 완료"
    else
        echo "❌ $method 실패"
    fi
    
    echo ""
done

# Cleanup temporary configs
rm -rf "$TMP_CONFIG_DIR"

echo "=========================================="
echo "모든 Baseline 실험 완료!"
echo "결과: $SAVE_DIR"
echo "=========================================="
