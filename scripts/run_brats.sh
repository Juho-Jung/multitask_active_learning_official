#!/bin/bash
# BraTS 2021 실험 실행 스크립트
# Tasks: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
# Inter-task correlation: Hierarchical (WT ⊃ TC ⊃ ET)

CONFIG_FILE="configs/brats_tumor_seg.json"
EXPERIMENT_NAME="brats_ctmi"
SAVE_DIR="./runs/mtl_results/brats"
DEVICE="cuda:1"

echo "=========================================="
echo "BraTS 2021 Benchmark (Medical Imaging)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Tasks: Whole Tumor + Tumor Core + Enhancing Tumor"
echo "Correlation: Hierarchical (WT ⊃ TC ⊃ ET)"
echo "Modalities: T1, T1ce, T2, FLAIR (4-channel MRI)"
echo "Expected CTMI improvement: +2.1%"
echo "=========================================="

python run_al.py \
    --config $CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --save_dir $SAVE_DIR \
    --device $DEVICE

echo "=========================================="
echo "BraTS 실험 완료!"
echo "=========================================="
