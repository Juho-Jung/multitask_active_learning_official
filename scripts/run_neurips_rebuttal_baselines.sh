#!/bin/bash
# NeurIPS 2026 rebuttal: reviewer-cited filter-then-diversify baselines (E1)
# 5 methods x 3 seeds = 15 runs. Usage on each server (repo must be synced first):
#   bash scripts/run_neurips_rebuttal_baselines.sh <config_name> <gpu_id>
# e.g. bash scripts/run_neurips_rebuttal_baselines.sh ms_ppal_s42 0

set -e
cd "$(dirname "$0")/.."

CONFIG_NAME=$1
GPU=${2:-0}
SAVE_ROOT=${SAVE_ROOT:-./runs/neurips_rebuttal_results}

if [ -z "$CONFIG_NAME" ]; then
  echo "Available configs:"
  ls configs/neurips_rebuttal/ | sed 's/\.json//'
  exit 1
fi

mkdir -p "$SAVE_ROOT/$CONFIG_NAME"
python3 -m multitask_active_learning.runner \
  --config "configs/neurips_rebuttal/${CONFIG_NAME}.json" \
  --save_dir "$SAVE_ROOT/$CONFIG_NAME" \
  --device "cuda:${GPU}" \
  2>&1 | tee "$SAVE_ROOT/$CONFIG_NAME/run.log"
