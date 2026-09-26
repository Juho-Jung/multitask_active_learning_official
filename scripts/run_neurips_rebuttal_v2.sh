#!/bin/bash
# NeurIPS rebuttal v2: main-table protocol (initial 1000 / batch 100 / 224x224 / 31 rounds).
# Usage: bash scripts/run_neurips_rebuttal_v2.sh <config_name> <gpu_id>
set -e
cd "$(dirname "$0")/.."

CONFIG_NAME=$1
GPU=${2:-0}
SAVE_ROOT=${SAVE_ROOT:-./runs/neurips_rebuttal_v2}

if [ -z "$CONFIG_NAME" ]; then
  echo "Available configs:"; ls configs/neurips_rebuttal_v2/ | sed 's/\.json//'; exit 1
fi

mkdir -p "$SAVE_ROOT/$CONFIG_NAME"
python3 -m multitask_active_learning.runner \
  --config "configs/neurips_rebuttal_v2/${CONFIG_NAME}.json" \
  --save_dir "$SAVE_ROOT/$CONFIG_NAME" \
  --device "cuda:${GPU}" \
  2>&1 | tee "$SAVE_ROOT/$CONFIG_NAME/run.log"
