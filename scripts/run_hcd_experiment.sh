#!/bin/bash
# Run the Hardness + Conflict + Diversity experiment

CONFIG_FILE="configs/coco/coco_hcd.json"
EXPERIMENT_NAME="hardness_conflict_diversity"
SAVE_DIR="./runs/mtl_results"
DEVICE="auto"

echo "=========================================="
echo "Hardness + Conflict + Diversity Experiment"
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
echo "Experiment completed!"
echo "=========================================="
