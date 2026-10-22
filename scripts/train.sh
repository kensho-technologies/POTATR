#!/usr/bin/env bash
# Copyright 2025-present Kensho Technologies, LLC.
set -euxo pipefail

# Example training invocation for POTATR on PubTables-v2, using all available GPUs.
# Training combines the model, loss and training configs.
#
# Usage:
#   bash scripts/train.sh [TRAIN_CONFIG] [CHECKPOINT_PATH] [SAVE_DIR]

MODEL_CONFIG="configs/model/potatr_v1.json"
LOSS_CONFIG="configs/loss/potatr_v1.json"
TRAIN_CONFIG="${1:-configs/train/pubtables_v2_stage1.json}"
CHECKPOINT="${2:-}"
SAVE_DIR="${3:-output/}"

EXTRA_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    EXTRA_ARGS="--model_load_path $CHECKPOINT --load_weights_only"
fi

uv run python -m potatr.train \
    --config_file "$MODEL_CONFIG" "$LOSS_CONFIG" "$TRAIN_CONFIG" \
    $EXTRA_ARGS \
    --model_save_dir "$SAVE_DIR" \
    --train_all_weights \
    --device cuda
