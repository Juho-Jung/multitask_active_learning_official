#!/bin/bash
# MTL 벤치마크 데이터셋 다운로드 스크립트

DATA_ROOT="./data"

echo "=========================================="
echo "MTL Benchmark Dataset Download Script"
echo "=========================================="

# 1. NYUv2 (가장 간단)
echo ""
echo "[1/3] NYUv2 Dataset"
echo "=========================================="
NYUV2_DIR="$DATA_ROOT/nyuv2"
mkdir -p $NYUV2_DIR

# 전처리된 버전 다운로드 (권장)
# 옵션 1: 공식 mat 파일
if [ ! -f "$NYUV2_DIR/nyu_depth_v2_labeled.mat" ]; then
    echo "Downloading NYUv2 labeled dataset..."
    wget -P $NYUV2_DIR http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
else
    echo "NYUv2 already exists."
fi

# 옵션 2: MTAN 전처리 버전 (더 사용하기 편함)
echo ""
echo "For preprocessed version, use:"
echo "  git clone https://github.com/lorenmt/mtan"
echo "  # Follow their data preparation instructions"

# 2. Cityscapes (가입 필요)
echo ""
echo "[2/3] Cityscapes Dataset"
echo "=========================================="
CITYSCAPES_DIR="$DATA_ROOT/cityscapes"
mkdir -p $CITYSCAPES_DIR

echo "Cityscapes requires registration at:"
echo "  https://www.cityscapes-dataset.com/"
echo ""
echo "After registration, download:"
echo "  - gtFine_trainvaltest.zip (annotations)"
echo "  - leftImg8bit_trainvaltest.zip (images)"
echo "  - disparity_trainvaltest.zip (depth)"
echo ""
echo "Or use cityscapesScripts:"
echo "  pip install cityscapesscripts"
echo "  export CITYSCAPES_DATASET=$CITYSCAPES_DIR"
echo "  csDownload gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip"

# 3. BraTS 2021 (가입 필요)
echo ""
echo "[3/3] BraTS 2021 Dataset"
echo "=========================================="
BRATS_DIR="$DATA_ROOT/brats2021"
mkdir -p $BRATS_DIR

echo "BraTS 2021 requires Synapse registration:"
echo "  https://www.synapse.org/#!Synapse:syn25829067"
echo ""
echo "After registration and data use agreement:"
echo "  pip install synapseclient"
echo "  synapse login"
echo "  synapse get syn25829067 -r --downloadLocation $BRATS_DIR"

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "  NYUv2:      $NYUV2_DIR (auto-download attempted)"
echo "  Cityscapes: $CITYSCAPES_DIR (manual registration needed)"
echo "  BraTS:      $BRATS_DIR (Synapse registration needed)"
echo "=========================================="
