#!/bin/bash
# NYUv2 2-Stage Selection 실험 실행 스크립트

CONFIG_FILE="configs/nyuv2_2stage_selection.json"
SAVE_DIR="./runs/mtl_results/$(date +%Y%m%d_%H%M%S)_nyuv2_2stage_selection"
DEVICE=${1:-"cuda:0"}

echo "=========================================="
echo "NYUv2 2-Stage Selection 실험 시작"
echo "=========================================="
echo "Config: nyuv2_2stage_selection.json"
echo ""
echo "핵심 변경사항:"
echo "  - Stage 1: Hardness + CTMI (Informativeness)"
echo "  - Stage 2: K-Center Greedy (Diversity)"
echo "  - Conflict 제거 (beta=0)"
echo "  - pool_factor=5 (5배수 후보군)"
echo ""
echo "Save dir: $SAVE_DIR"
echo "Device: $DEVICE"
echo "=========================================="

python -m multitask_active_learning.runner \
    --config $CONFIG_FILE \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "실험 완료!"
echo "결과: $SAVE_DIR"
echo "=========================================="
