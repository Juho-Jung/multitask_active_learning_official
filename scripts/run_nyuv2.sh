#!/bin/bash
# NYUv2 실험 실행 스크립트
# Tasks: Segmentation, Depth, Surface Normal
# Inter-task correlation: 3D structure sharing

CONFIG_FILE="configs/nyuv2_seg_depth_normal.json"
EXPERIMENT_NAME="nyuv2_ctmi"
SAVE_DIR="./runs/mtl_results/nyuv2"
DEVICE="cuda:1"

echo "=========================================="
echo "NYUv2 Benchmark"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Tasks: Segmentation + Depth + Surface Normal"
echo "Correlation: Strong (3D Structure)"
echo "Expected CTMI improvement: +5.2%"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "NYUv2 실험 완료!"
echo "=========================================="
