#!/bin/bash
# Linear Scalarization vs ODS (2-Stage)
# 
# Purpose: Show that naively combining Hardness + CTMI + Diversity via
# weighted sum suffers from diversity dilution compared to ODS's
# orthogonal 2-stage decomposition.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Linear Scalarization vs ODS"
echo "============================================"

# COCO experiment
echo ""
echo "[1/3] Running COCO: Linear Scalarization vs ODS..."
python "$PROJECT_DIR/run_al.py" \
    --config "$PROJECT_DIR/configs/coco/coco_linear_vs_ods.json"

# NYUv2 experiment (if config exists)
if [ -f "$PROJECT_DIR/configs/nyuv2/nyuv2_linear_vs_ods.json" ]; then
    echo ""
    echo "[2/3] Running NYUv2: Linear Scalarization vs ODS..."
    python "$PROJECT_DIR/run_al.py" \
        --config "$PROJECT_DIR/configs/nyuv2/nyuv2_linear_vs_ods.json"
fi

# Cityscapes experiment (if config exists)
if [ -f "$PROJECT_DIR/configs/cityscapes/cityscapes_linear_vs_ods.json" ]; then
    echo ""
    echo "[3/3] Running Cityscapes: Linear Scalarization vs ODS..."
    python "$PROJECT_DIR/run_al.py" \
        --config "$PROJECT_DIR/configs/cityscapes/cityscapes_linear_vs_ods.json"
fi

echo ""
echo "============================================"
echo " Experiment complete!"
echo " Results saved to: results/linear_vs_ods/"
echo "============================================"
