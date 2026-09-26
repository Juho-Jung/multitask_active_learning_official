#!/bin/bash
# Hardness + Conflict + Diversity 실험 실행 스크립트

CONFIG_FILE="configs/coco_hcd.json"
EXPERIMENT_NAME="hardness_conflict_diversity"
SAVE_DIR="./runs/mtl_results"
DEVICE="auto"

echo "=========================================="
echo "Hardness + Conflict + Diversity 실험"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Save Dir: $SAVE_DIR"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "실험 완료!"
echo "=========================================="
