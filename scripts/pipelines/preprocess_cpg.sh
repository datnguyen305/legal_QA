#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/QA}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/train_data.json}
DEV_DATA=${DEV_DATA:-$DATA_DIR/dev_data.json}
CONTEXT_DIR=${CONTEXT_DIR:-$REPO_ROOT/dataset/contexts}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
DEV_LIMIT=${DEV_LIMIT:-}

CPG_CACHE_DIR=${CPG_CACHE_DIR:-cache/cpg}
CPG_REBUILD_CACHE=${CPG_REBUILD_CACHE:-0}
CPG_EPOCHS=${CPG_EPOCHS:-20}
CPG_CHUNK_SIZES=${CPG_CHUNK_SIZES:-50,100,200,500}
CPG_MAX_CONTEXT_TOKENS=${CPG_MAX_CONTEXT_TOKENS:-1200}
CPG_EASY_RATIO=${CPG_EASY_RATIO:-1.0}
CPG_EASY_RATIO_DECAY=${CPG_EASY_RATIO_DECAY:-0.25}

limit_args() {
  local name=$1
  local value=$2
  if [[ -n "$value" ]]; then
    printf -- "--%s %s" "$name" "$value"
  fi
}

rebuild_args=()
if [[ "$CPG_REBUILD_CACHE" == "1" ]]; then
  rebuild_args+=(--rebuild-cache)
fi

python3 scripts/preprocess_cpg.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --cache-dir "$CPG_CACHE_DIR" \
  --epochs "$CPG_EPOCHS" \
  --chunk-sizes "$CPG_CHUNK_SIZES" \
  --max-context-tokens "$CPG_MAX_CONTEXT_TOKENS" \
  --easy-ratio "$CPG_EASY_RATIO" \
  --easy-ratio-decay "$CPG_EASY_RATIO_DECAY" \
  "${rebuild_args[@]}" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")
