#!/bin/bash
# Run the Cityscapes experiment
# Tasks: Segmentation, Depth, Instance Segmentation
# Inter-task correlation: Scene geometry sharing (strong correlation)

CONFIG_FILE="configs/cityscapes/cityscapes_seg_depth_inst.json"
EXPERIMENT_NAME="cityscapes_ctmi"
SAVE_DIR="./runs/mtl_results/cityscapes"
DEVICE=${1:-"cuda:0"}

echo "=========================================="
echo "Cityscapes Benchmark"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Tasks: Segmentation + Depth + Instance"
echo "Correlation: Strong (Scene Geometry)"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "Cityscapes experiment completed!"
echo "=========================================="
