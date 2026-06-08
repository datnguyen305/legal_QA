#!/usr/bin/env bash
set -euo pipefail

# EQUALS proposed model pipeline: retrieval + BERT-style MRC.
# Override any setting by prefixing the command, for example:
#   LIMIT=100 RETRIEVER=bm25 bash scripts/run_equals_pipeline.sh

CONFIG="${CONFIG:-configs/models/equals.json}"
eval "$(python3 scripts/pipelines/config_env.py "$CONFIG")"

TRAIN_DATA="${TRAIN_DATA:-dataset/train_data.json}"
DEV_DATA="${DEV_DATA:-dataset/dev_data.json}"
TEST_DATA="${TEST_DATA:-dataset/test_data.json}"
CONTEXT_DIR="${CONTEXT_DIR:-dataset/contexts}"
BASE_MODEL="${BASE_MODEL:-bert-base-multilingual-cased}"
MODEL_DIR="${MODEL_DIR:-models/equals_mrc}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RETRIEVER="${RETRIEVER:-gold}" # gold, bm25, or sbert
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
TOP_K="${TOP_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_ANSWER_LENGTH="${MAX_ANSWER_LENGTH:-261}"
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
  --epochs "$EPOCHS"
  --lr "$LR"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-length "$MAX_LENGTH"
)
if [[ -n "$TRAIN_LIMIT" ]]; then
  train_args+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$DEV_LIMIT" ]]; then
  train_args+=(--dev-limit "$DEV_LIMIT")
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 scripts/train_equals_mrc.py "${train_args[@]}"
fi

predictions="${PREDICTIONS:-$OUTPUT_DIR/equals_${RETRIEVER}_mrc.jsonl}"
metrics="${METRICS:-$OUTPUT_DIR/equals_${RETRIEVER}_mrc_metrics.json}"

run_args=(
  --retriever "$RETRIEVER"
  --qa-model "$MODEL_DIR"
  --data "$TEST_DATA"
  --context-dir "$CONTEXT_DIR"
  --top-k "$TOP_K"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-length "$MAX_LENGTH"
  --max-answer-length "$MAX_ANSWER_LENGTH"
  --batch-size "$BATCH_SIZE"
  --output "$predictions"
)
if [[ "$RETRIEVER" != "gold" ]]; then
  run_args+=(--corpus-data "$TRAIN_DATA" "$DEV_DATA" "$TEST_DATA")
fi
if [[ "$RETRIEVER" == "sbert" ]]; then
  run_args+=(--sbert-model "$SBERT_MODEL")
fi
if [[ -n "$LIMIT" ]]; then
  run_args+=(--limit "$LIMIT")
fi
if [[ -n "$DEVICE" ]]; then
  run_args+=(--device "$DEVICE")
fi

python3 scripts/run_equals.py "${run_args[@]}"
eval_args=(--predictions "$predictions" --output "$metrics")
if [[ "${EVAL_UPPER_BOUND:-0}" == "1" ]]; then
  eval_args+=(--upper-bound)
fi
python3 scripts/evaluate_predictions.py "${eval_args[@]}"

echo "Predictions: $predictions"
echo "Metrics: $metrics"
