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

MODEL_DIR=${MODEL_DIR:-models/latentqa}
PREDICTIONS=${PREDICTIONS:-outputs/latentqa_predictions.jsonl}
METRICS=${METRICS:-outputs/latentqa_metrics.json}
LATENTQA_BATCH_SIZE=${LATENTQA_BATCH_SIZE:-64}
LATENTQA_EPOCHS=${LATENTQA_EPOCHS:-20}
LATENTQA_PATIENCE=${LATENTQA_PATIENCE:-3}
LATENTQA_MAX_CONTEXT_TOKENS=${LATENTQA_MAX_CONTEXT_TOKENS:-800}
LATENTQA_MAX_ANSWER_TOKENS=${LATENTQA_MAX_ANSWER_TOKENS:-96}

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

python3 scripts/train_latentqa.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$LATENTQA_BATCH_SIZE" \
  --epochs "$LATENTQA_EPOCHS" \
  --patience "$LATENTQA_PATIENCE" \
  --max-context-tokens "$LATENTQA_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$LATENTQA_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 scripts/run_latentqa.py \
  --model-dir "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output "$PREDICTIONS" \
  $(limit_args limit "$TEST_LIMIT")

python3 scripts/evaluate_predictions.py \
  --predictions "$PREDICTIONS" \
  --output "$METRICS" \
  "${bert_args[@]}"
