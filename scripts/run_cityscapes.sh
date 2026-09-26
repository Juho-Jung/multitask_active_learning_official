#!/bin/bash
# Cityscapes 실험 실행 스크립트
# Tasks: Segmentation, Depth, Instance Segmentation
# Inter-task correlation: Scene geometry sharing (strong correlation)

CONFIG_FILE="configs/cityscapes_seg_depth_inst.json"
EXPERIMENT_NAME="cityscapes_ctmi"
SAVE_DIR="./runs/mtl_results/cityscapes"
DEVICE="cuda:1"

echo "=========================================="
echo "Cityscapes Benchmark"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Tasks: Segmentation + Depth + Instance"
echo "Correlation: Strong (Scene Geometry)"
echo "Expected CTMI improvement: +4.5%"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "Cityscapes 실험 완료!"
echo "=========================================="
