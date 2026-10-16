#!/bin/bash
# Run the BraTS 2021 experiment
# Tasks: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
# Inter-task correlation: Hierarchical (WT ⊃ TC ⊃ ET)

CONFIG_FILE="configs/brats_tumor_seg.json"
EXPERIMENT_NAME="brats_ctmi"
SAVE_DIR="./runs/mtl_results/brats"
DEVICE=${1:-"cuda:0"}

echo "=========================================="
echo "BraTS 2021 Benchmark (Medical Imaging)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Tasks: Whole Tumor + Tumor Core + Enhancing Tumor"
echo "Correlation: Hierarchical (WT ⊃ TC ⊃ ET)"
echo "Modalities: T1, T1ce, T2, FLAIR (4-channel MRI)"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "BraTS experiment completed!"
echo "=========================================="
