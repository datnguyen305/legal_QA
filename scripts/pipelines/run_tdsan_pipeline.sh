#!/usr/bin/env bash
set -euo pipefail

# Token-level Dynamic Self-Attention Network / DynSAN pipeline.
# Smoke run:
#   LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/run_tdsan_pipeline.sh

CONFIG="${CONFIG:-configs/models/tdsan.json}"
eval "$(python3 scripts/pipelines/config_env.py "$CONFIG")"

TRAIN_DATA="${TRAIN_DATA:-dataset/train_data.json}"
DEV_DATA="${DEV_DATA:-dataset/dev_data.json}"
TEST_DATA="${TEST_DATA:-dataset/test_data.json}"
CONTEXT_DIR="${CONTEXT_DIR:-dataset/contexts}"
MODEL_DIR="${MODEL_DIR:-models/tdsan}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-3}"
MAX_PASSAGES="${MAX_PASSAGES:-4}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_QUESTION_TOKENS="${MAX_QUESTION_TOKENS:-96}"
MAX_PASSAGE_TOKENS="${MAX_PASSAGE_TOKENS:-1536}"
HIDDEN="${HIDDEN:-128}"
HEADS="${HEADS:-8}"
TOP_K="${TOP_K:-256}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
DEV_LIMIT="${DEV_LIMIT:-}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

train_args=(
  --train-data "$TRAIN_DATA"
  --dev-data "$DEV_DATA"
  --context-dir "$CONTEXT_DIR"
  --output-dir "$MODEL_DIR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --max-passages "$MAX_PASSAGES"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-question-tokens "$MAX_QUESTION_TOKENS"
  --max-passage-tokens "$MAX_PASSAGE_TOKENS"
  --hidden "$HIDDEN"
  --heads "$HEADS"
  --top-k "$TOP_K"
  --num-workers "$NUM_WORKERS"
)
if [[ -n "$TRAIN_LIMIT" ]]; then train_args+=(--train-limit "$TRAIN_LIMIT"); fi
if [[ -n "$DEV_LIMIT" ]]; then train_args+=(--dev-limit "$DEV_LIMIT"); fi
if [[ -n "$DEVICE" ]]; then train_args+=(--device "$DEVICE"); fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 scripts/train_tdsan.py "${train_args[@]}"
fi

predictions="${PREDICTIONS:-$OUTPUT_DIR/tdsan.jsonl}"
metrics="${METRICS:-$OUTPUT_DIR/tdsan_metrics.json}"
run_args=(
  --model-dir "$MODEL_DIR"
  --data "$TEST_DATA"
  --context-dir "$CONTEXT_DIR"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --output "$predictions"
)
if [[ -n "$LIMIT" ]]; then run_args+=(--limit "$LIMIT"); fi
if [[ -n "$DEVICE" ]]; then run_args+=(--device "$DEVICE"); fi

python3 scripts/run_tdsan.py "${run_args[@]}"
eval_args=(--predictions "$predictions" --output "$metrics")
if [[ "${EVAL_UPPER_BOUND:-0}" == "1" ]]; then
  eval_args+=(--upper-bound)
fi
python3 scripts/evaluate_predictions.py "${eval_args[@]}"

echo "Predictions: $predictions"
echo "Metrics: $metrics"
