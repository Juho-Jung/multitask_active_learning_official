#!/bin/bash
# Unified protocol (initial 1000 / batch 100 / 224x224 / 31 rounds) for ODS and the
# filter-then-diversify baselines.
# Usage: bash scripts/run_unified_protocol.sh <config_name> <gpu_id>
# e.g.   bash scripts/run_unified_protocol.sh mt_ppal_s42 0
set -e
cd "$(dirname "$0")/.."

CONFIG_NAME=$1
GPU=${2:-0}
SAVE_ROOT=${SAVE_ROOT:-./runs/unified_protocol}

if [ -z "$CONFIG_NAME" ]; then
  echo "Available configs:"; ls configs/unified_protocol/ | sed 's/\.json//'; exit 1
fi

mkdir -p "$SAVE_ROOT/$CONFIG_NAME"
python3 -m multitask_active_learning.runner \
  --config "configs/unified_protocol/${CONFIG_NAME}.json" \
  --save_dir "$SAVE_ROOT/$CONFIG_NAME" \
  --device "cuda:${GPU}" \
  2>&1 | tee "$SAVE_ROOT/$CONFIG_NAME/run.log"
