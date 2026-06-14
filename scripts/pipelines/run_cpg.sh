#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-dataset}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/train_data.json}
DEV_DATA=${DEV_DATA:-$DATA_DIR/dev_data.json}
TEST_DATA=${TEST_DATA:-$DATA_DIR/test_data.json}
CONTEXT_DIR=${CONTEXT_DIR:-$DATA_DIR/contexts}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
DEV_LIMIT=${DEV_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
BERTSCORE=${BERTSCORE:-1}
BERTSCORE_MODEL=${BERTSCORE_MODEL:-bert-base-multilingual-cased}
BERTSCORE_BATCH_SIZE=${BERTSCORE_BATCH_SIZE:-16}
AMP=${AMP:-bf16}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}
BERTSCORE_DEVICE=${BERTSCORE_DEVICE:-$DEVICE}

MODEL_DIR=${MODEL_DIR:-models/cpg}
PREDICTIONS=${PREDICTIONS:-outputs/cpg_predictions.jsonl}
METRICS=${METRICS:-outputs/cpg_metrics.json}
CPG_CACHE_DIR=${CPG_CACHE_DIR:-cache/cpg}
CPG_DISK_CACHE=${CPG_DISK_CACHE:-1}
CPG_REBUILD_CACHE=${CPG_REBUILD_CACHE:-0}
CPG_BATCH_SIZE=${CPG_BATCH_SIZE:-16}
CPG_EPOCHS=${CPG_EPOCHS:-20}
CPG_PATIENCE=${CPG_PATIENCE:-3}
CPG_CHUNK_SIZES=${CPG_CHUNK_SIZES:-50,100,200,500}
CPG_MAX_CONTEXT_TOKENS=${CPG_MAX_CONTEXT_TOKENS:-1200}
CPG_MAX_ANSWER_TOKENS=${CPG_MAX_ANSWER_TOKENS:-96}

limit_args() {
  local name=$1
  local value=$2
  if [[ -n "$value" ]]; then
    printf -- "--%s %s" "$name" "$value"
  fi
}

bert_args=(--bertscore-model "$BERTSCORE_MODEL" --bertscore-batch-size "$BERTSCORE_BATCH_SIZE")
if [[ -n "$BERTSCORE_DEVICE" ]]; then
  bert_args+=(--bertscore-device "$BERTSCORE_DEVICE")
fi
if [[ "$BERTSCORE" == "0" ]]; then
  bert_args=(--no-bertscore)
fi

cache_args=(--cache-dir "$CPG_CACHE_DIR")
if [[ "$CPG_DISK_CACHE" == "0" ]]; then
  cache_args+=(--no-disk-cache)
fi
if [[ "$CPG_REBUILD_CACHE" == "1" ]]; then
  cache_args+=(--rebuild-cache)
fi

python3 scripts/train_cpg.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$CPG_BATCH_SIZE" \
  --epochs "$CPG_EPOCHS" \
  --patience "$CPG_PATIENCE" \
  --chunk-sizes "$CPG_CHUNK_SIZES" \
  --max-context-tokens "$CPG_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$CPG_MAX_ANSWER_TOKENS" \
  "${cache_args[@]}" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 scripts/run_cpg.py \
  --model-dir "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output "$PREDICTIONS" \
  $(limit_args limit "$TEST_LIMIT")

python3 scripts/evaluate_predictions.py \
  --predictions "$PREDICTIONS" \
  --output "$METRICS" \
  "${bert_args[@]}"
