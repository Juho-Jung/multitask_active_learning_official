#!/bin/bash
# Run all multi-task benchmarks

SAVE_DIR="./runs/mtl_results"
DEVICE="cuda:0"

echo "=========================================="
echo "Multi-Task Active Learning Benchmark Suite"
echo "=========================================="

# 1. COCO (Vision - Classification, Detection, Segmentation)
echo ""
echo "[1/4] COCO Benchmark"
echo "Tasks: Classification, Detection, Segmentation"
echo "=========================================="
python run_al.py \
    --config configs/coco/coco_improved_ctmi.json \
    --experiment_name coco_ctmi_benchmark \
    --save_dir $SAVE_DIR/coco \
    --device $DEVICE

# 2. Cityscapes (Autonomous Driving - Seg, Depth, Instance)
echo ""
echo "[2/4] Cityscapes Benchmark"
echo "Tasks: Segmentation, Depth, Instance Segmentation"
echo "Inter-task correlation: Scene geometry sharing"
echo "=========================================="
python run_al.py \
    --config configs/cityscapes/cityscapes_seg_depth_inst.json \
    --experiment_name cityscapes_ctmi_benchmark \
    --save_dir $SAVE_DIR/cityscapes \
    --device $DEVICE

# 3. NYUv2 (Indoor Scenes - Seg, Depth, Normal)
echo ""
echo "[3/4] NYUv2 Benchmark"
echo "Tasks: Segmentation, Depth, Surface Normal"
echo "Inter-task correlation: 3D structure sharing"
echo "=========================================="
python run_al.py \
    --config configs/nyuv2/nyuv2_seg_depth_normal.json \
    --experiment_name nyuv2_ctmi_benchmark \
    --save_dir $SAVE_DIR/nyuv2 \
    --device $DEVICE

# 4. BraTS (Medical Imaging - Hierarchical Tumor Segmentation)
echo ""
echo "[4/4] BraTS 2021 Benchmark"
echo "Tasks: Whole Tumor, Tumor Core, Enhancing Tumor"
echo "Inter-task correlation: Hierarchical (WT ⊃ TC ⊃ ET)"
echo "=========================================="
python run_al.py \
    --config configs/brats_tumor_seg.json \
    --experiment_name brats_ctmi_benchmark \
    --save_dir $SAVE_DIR/brats \
    --device $DEVICE

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved to: $SAVE_DIR"
echo "=========================================="
