#!/bin/bash
# Multi-seed runs (seeds 42 / 123 / 456) and score-normalization ablations.
# Usage:
#   bash scripts/run_multiseed.sh <config_name> <gpu_id>
# e.g. bash scripts/run_multiseed.sh ms_ppal_s42 0
#      bash scripts/run_multiseed.sh ../ablations/score_norm_rank_s42 0

set -e
cd "$(dirname "$0")/.."

CONFIG_NAME=$1
GPU=${2:-0}
SAVE_ROOT=${SAVE_ROOT:-./runs/multiseed}

if [ -z "$CONFIG_NAME" ]; then
  echo "Available configs:"
  ls configs/multiseed/ | sed 's/\.json//'
  exit 1
fi

mkdir -p "$SAVE_ROOT/$CONFIG_NAME"
python3 -m multitask_active_learning.runner \
  --config "configs/multiseed/${CONFIG_NAME}.json" \
  --save_dir "$SAVE_ROOT/$CONFIG_NAME" \
  --device "cuda:${GPU}" \
  2>&1 | tee "$SAVE_ROOT/$CONFIG_NAME/run.log"
