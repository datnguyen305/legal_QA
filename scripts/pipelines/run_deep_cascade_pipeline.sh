#!/usr/bin/env bash
set -euo pipefail

# Deep Cascade Model pipeline.
# Smoke run:
#   LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/run_deep_cascade_pipeline.sh

CONFIG="${CONFIG:-configs/models/deep_cascade.json}"
eval "$(python3 scripts/pipelines/config_env.py "$CONFIG")"

TRAIN_DATA="${TRAIN_DATA:-dataset/train_data.json}"
DEV_DATA="${DEV_DATA:-dataset/dev_data.json}"
TEST_DATA="${TEST_DATA:-dataset/test_data.json}"
CONTEXT_DIR="${CONTEXT_DIR:-dataset/contexts}"
MODEL_DIR="${MODEL_DIR:-models/deep_cascade}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
RANKER_EPOCHS="${RANKER_EPOCHS:-1}"
LR="${LR:-5e-4}"
MAX_DOCS="${MAX_DOCS:-4}"
MAX_PARAGRAPHS="${MAX_PARAGRAPHS:-2}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_QUESTION_TOKENS="${MAX_QUESTION_TOKENS:-96}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-1536}"
HIDDEN="${HIDDEN:-128}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
DEV_LIMIT="${DEV_LIMIT:-}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

train_args=(
  --train-data "$TRAIN_DATA"
  --dev-data "$DEV_DATA"
  --context-dir "$CONTEXT_DIR"
  --output-dir "$MODEL_DIR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --ranker-epochs "$RANKER_EPOCHS"
  --lr "$LR"
  --max-docs "$MAX_DOCS"
  --max-paragraphs "$MAX_PARAGRAPHS"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-question-tokens "$MAX_QUESTION_TOKENS"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
  --hidden "$HIDDEN"
)
if [[ -n "$TRAIN_LIMIT" ]]; then train_args+=(--train-limit "$TRAIN_LIMIT"); fi
if [[ -n "$DEV_LIMIT" ]]; then train_args+=(--dev-limit "$DEV_LIMIT"); fi
if [[ -n "$DEVICE" ]]; then train_args+=(--device "$DEVICE"); fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 scripts/train_deep_cascade.py "${train_args[@]}"
fi

predictions="${PREDICTIONS:-$OUTPUT_DIR/deep_cascade.jsonl}"
metrics="${METRICS:-$OUTPUT_DIR/deep_cascade_metrics.json}"
run_args=(
  --model-dir "$MODEL_DIR"
  --data "$TEST_DATA"
  --context-dir "$CONTEXT_DIR"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --output "$predictions"
)
if [[ -n "$LIMIT" ]]; then run_args+=(--limit "$LIMIT"); fi
if [[ -n "$DEVICE" ]]; then run_args+=(--device "$DEVICE"); fi

python3 scripts/run_deep_cascade.py "${run_args[@]}"
eval_args=(--predictions "$predictions" --output "$metrics")
if [[ "${EVAL_UPPER_BOUND:-0}" == "1" ]]; then
  eval_args+=(--upper-bound)
fi
python3 scripts/evaluate_predictions.py "${eval_args[@]}"

echo "Predictions: $predictions"
echo "Metrics: $metrics"
