#!/usr/bin/env bash
set -euo pipefail

# Select, Answer and Explain proposed model pipeline.
# Example smoke run:
#   LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/run_sae_pipeline.sh

TRAIN_DATA="${TRAIN_DATA:-dataset/train_data.json}"
DEV_DATA="${DEV_DATA:-dataset/dev_data.json}"
TEST_DATA="${TEST_DATA:-dataset/test_data.json}"
CONTEXT_DIR="${CONTEXT_DIR:-dataset/contexts}"
BASE_MODEL="${BASE_MODEL:-bert-base-multilingual-cased}"
MODEL_DIR="${MODEL_DIR:-models/sae}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SELECTOR_EPOCHS="${SELECTOR_EPOCHS:-1}"
ANSWER_EPOCHS="${ANSWER_EPOCHS:-2}"
LR="${LR:-2e-5}"
MAX_DOCS="${MAX_DOCS:-6}"
TOP_K="${TOP_K:-2}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_SENTENCES="${MAX_SENTENCES:-96}"
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
  --base-model "$BASE_MODEL"
  --output-dir "$MODEL_DIR"
  --batch-size "$BATCH_SIZE"
  --selector-epochs "$SELECTOR_EPOCHS"
  --answer-epochs "$ANSWER_EPOCHS"
  --lr "$LR"
  --max-docs "$MAX_DOCS"
  --top-k "$TOP_K"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-length "$MAX_LENGTH"
  --max-sentences "$MAX_SENTENCES"
)
if [[ -n "$TRAIN_LIMIT" ]]; then
  train_args+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$DEV_LIMIT" ]]; then
  train_args+=(--dev-limit "$DEV_LIMIT")
fi
if [[ -n "$DEVICE" ]]; then
  train_args+=(--device "$DEVICE")
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 scripts/train_sae.py "${train_args[@]}"
fi

predictions="$OUTPUT_DIR/sae.jsonl"
metrics="$OUTPUT_DIR/sae_metrics.json"
run_args=(
  --model-dir "$MODEL_DIR"
  --data "$TEST_DATA"
  --corpus-data "$TRAIN_DATA" "$DEV_DATA" "$TEST_DATA"
  --context-dir "$CONTEXT_DIR"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --output "$predictions"
)
if [[ -n "$LIMIT" ]]; then
  run_args+=(--limit "$LIMIT")
fi
if [[ -n "$DEVICE" ]]; then
  run_args+=(--device "$DEVICE")
fi

python3 scripts/run_sae.py "${run_args[@]}"
eval_args=(--predictions "$predictions" --output "$metrics")
if [[ "${EVAL_UPPER_BOUND:-0}" == "1" ]]; then
  eval_args+=(--upper-bound)
fi
python3 scripts/evaluate_predictions.py "${eval_args[@]}"

echo "Predictions: $predictions"
echo "Metrics: $metrics"
