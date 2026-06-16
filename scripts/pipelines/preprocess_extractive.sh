#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-dataset}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/train_data.json}
DEV_DATA=${DEV_DATA:-$DATA_DIR/dev_data.json}
CONTEXT_DIR=${CONTEXT_DIR:-$DATA_DIR/contexts}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
DEV_LIMIT=${DEV_LIMIT:-}

EXTRACTIVE_CACHE_DIR=${EXTRACTIVE_CACHE_DIR:-cache/extractive}
EXTRACTIVE_REBUILD_CACHE=${EXTRACTIVE_REBUILD_CACHE:-0}
EXTRACTIVE_DISK_CACHE=${EXTRACTIVE_DISK_CACHE:-1}
EXTRACTIVE_MAX_PASSAGES=${EXTRACTIVE_MAX_PASSAGES:-6}
EXTRACTIVE_PASSAGE_LEN=${EXTRACTIVE_PASSAGE_LEN:-256}
EXTRACTIVE_MAX_CONTEXT_CHARS=${EXTRACTIVE_MAX_CONTEXT_CHARS:-12000}

limit_args() {
  local name=$1
  local value=$2
  if [[ -n "$value" ]]; then
    printf -- "--%s %s" "$name" "$value"
  fi
}

cache_args=(--cache-dir "$EXTRACTIVE_CACHE_DIR")
if [[ "$EXTRACTIVE_DISK_CACHE" == "0" ]]; then
  cache_args+=(--no-disk-cache)
fi
if [[ "$EXTRACTIVE_REBUILD_CACHE" == "1" ]]; then
  cache_args+=(--rebuild-cache)
fi

python3 scripts/preprocess_extractive.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --max-context-chars "$EXTRACTIVE_MAX_CONTEXT_CHARS" \
  --max-passages "$EXTRACTIVE_MAX_PASSAGES" \
  --passage-len "$EXTRACTIVE_PASSAGE_LEN" \
  "${cache_args[@]}" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")
