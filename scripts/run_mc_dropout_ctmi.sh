#!/bin/bash
# MC-Dropout CTMI 실험 실행 스크립트
# Conflict 제거, Hardness + MC-Dropout CTMI만 사용

cd /opt/multitask_active_learning

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"

# 결과 저장 경로
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./runs/mtl_results/${TIMESTAMP}_mc_dropout_ctmi"
mkdir -p $SAVE_DIR

echo "=============================================="
echo "MC-Dropout CTMI 실험 시작"
echo "=============================================="
echo "Config: coco_mc_dropout_ctmi.json"
echo "특징:"
echo "  - Conflict 제거 (beta=0)"
echo "  - MC-Dropout CTMI 사용 (K=10 samples)"
echo "  - Hardness + CTMI 조합"
echo "  - Diversity δ=0.5"
echo "Save dir: $SAVE_DIR"
echo "=============================================="

# 실험 실행
python -m multitask_active_learning.runner \
    --config configs/coco_mc_dropout_ctmi.json \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    2>&1 | tee $SAVE_DIR/run.log

echo "=============================================="
echo "실험 완료!"
echo "결과: $SAVE_DIR"
echo "=============================================="
