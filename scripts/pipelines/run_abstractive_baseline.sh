#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
MODEL=${MODEL:-dcmn_plus}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/QA}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/train_data.json}
DEV_DATA=${DEV_DATA:-$DATA_DIR/dev_data.json}
TEST_DATA=${TEST_DATA:-$DATA_DIR/test_data.json}
CONTEXT_DIR=${CONTEXT_DIR:-$REPO_ROOT/dataset/contexts}
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

MODEL_DIR=${MODEL_DIR:-models/$MODEL}
PREDICTIONS=${PREDICTIONS:-outputs/${MODEL}_predictions.jsonl}
METRICS=${METRICS:-outputs/${MODEL}_metrics.json}
ABSTRACTIVE_BATCH_SIZE=${ABSTRACTIVE_BATCH_SIZE:-64}
ABSTRACTIVE_EPOCHS=${ABSTRACTIVE_EPOCHS:-20}
ABSTRACTIVE_PATIENCE=${ABSTRACTIVE_PATIENCE:-3}
ABSTRACTIVE_MAX_CONTEXT_TOKENS=${ABSTRACTIVE_MAX_CONTEXT_TOKENS:-800}
ABSTRACTIVE_MAX_ANSWER_TOKENS=${ABSTRACTIVE_MAX_ANSWER_TOKENS:-96}

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

python3 scripts/train_abstractive_baseline.py \
  --model "$MODEL" \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$ABSTRACTIVE_BATCH_SIZE" \
  --epochs "$ABSTRACTIVE_EPOCHS" \
  --patience "$ABSTRACTIVE_PATIENCE" \
  --max-context-tokens "$ABSTRACTIVE_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$ABSTRACTIVE_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 scripts/run_abstractive_baseline.py \
  --model-dir "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output "$PREDICTIONS" \
  --device "$DEVICE" \
  $(limit_args limit "$TEST_LIMIT")

python3 scripts/evaluate_predictions.py \
  --predictions "$PREDICTIONS" \
  --output "$METRICS" \
  "${bert_args[@]}"
