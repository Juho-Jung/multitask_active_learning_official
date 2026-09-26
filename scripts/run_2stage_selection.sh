#!/bin/bash
# 2-Stage Selection 실험 실행 스크립트
# 
# 핵심 개선:
# 1. Diversity 희석 해결: K-Center Greedy로 2단계 분리
# 2. Conflict 역설 해결: Conflict 제외 (beta=0)
# 3. 연산 효율: 후보군(pool_factor=5)에만 Diversity 계산

cd /opt/multitask_active_learning

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"

# 결과 저장 경로
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="./runs/mtl_results/${TIMESTAMP}_2stage_selection"
mkdir -p $SAVE_DIR

echo "=============================================="
echo "2-Stage Selection 실험 시작"
echo "=============================================="
echo "Config: coco_2stage_selection.json"
echo ""
echo "핵심 변경사항:"
echo "  - Stage 1: Hardness + CTMI (Informativeness)"
echo "  - Stage 2: K-Center Greedy (Diversity)"
echo "  - Conflict 제거 (beta=0)"
echo "  - pool_factor=5 (5배수 후보군)"
echo ""
echo "Save dir: $SAVE_DIR"
echo "=============================================="

# 실험 실행
python -m multitask_active_learning.runner \
    --config configs/coco_2stage_selection.json \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    2>&1 | tee $SAVE_DIR/run.log

echo "=============================================="
echo "실험 완료!"
echo "결과: $SAVE_DIR"
echo "=============================================="
