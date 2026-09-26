#!/bin/bash
# Improved CTMI 실험 실행 스크립트

CONFIG_FILE="configs/coco_improved_ctmi.json"
EXPERIMENT_NAME="improved_ctmi_covariance"
SAVE_DIR="./runs/mtl_results"
DEVICE="auto"

echo "=========================================="
echo "Improved CTMI (Covariance-based) 실험"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Save Dir: $SAVE_DIR"
echo ""
echo "Key changes:"
echo "  - Covariance-based MI approximation"
echo "  - Low MI -> High acquisition score (inverted)"
echo "  - Optimal hyperparameters: α=0.5, β=0.5, γ=1.5, δ=0.3"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device cuda:1

echo "=========================================="
echo "실험 완료!"
echo "=========================================="
