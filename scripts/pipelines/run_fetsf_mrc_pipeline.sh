#!/usr/bin/env bash
set -euo pipefail

# FETSF-MRC proposed model pipeline.
# Override any setting by prefixing the command, for example:
#   LIMIT=100 TRAIN_LIMIT=1000 bash scripts/run_fetsf_mrc_pipeline.sh

CONFIG="${CONFIG:-configs/models/fetsf_mrc.json}"
eval "$(python3 scripts/pipelines/config_env.py "$CONFIG")"

TRAIN_DATA="${TRAIN_DATA:-dataset/train_data.json}"
DEV_DATA="${DEV_DATA:-dataset/dev_data.json}"
TEST_DATA="${TEST_DATA:-dataset/test_data.json}"
CONTEXT_DIR="${CONTEXT_DIR:-dataset/contexts}"
BASE_MODEL="${BASE_MODEL:-bert-base-multilingual-cased}"
MODEL_DIR="${MODEL_DIR:-models/fetsf_mrc}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_SENTENCES="${MAX_SENTENCES:-64}"
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
  --base-model "$BASE_MODEL"
  --output-dir "$MODEL_DIR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --max-context-chars "$MAX_CONTEXT_CHARS"
  --max-length "$MAX_LENGTH"
  --max-sentences "$MAX_SENTENCES"
  --num-workers "$NUM_WORKERS"
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
  python3 scripts/train_fetsf_mrc.py "${train_args[@]}"
fi

predictions="${PREDICTIONS:-$OUTPUT_DIR/fetsf_mrc.jsonl}"
metrics="${METRICS:-$OUTPUT_DIR/fetsf_mrc_metrics.json}"

run_args=(
  --model-dir "$MODEL_DIR"
  --data "$TEST_DATA"
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

python3 scripts/run_fetsf_mrc.py "${run_args[@]}"
eval_args=(--predictions "$predictions" --output "$metrics")
if [[ "${EVAL_UPPER_BOUND:-0}" == "1" ]]; then
  eval_args+=(--upper-bound)
fi
if [[ "${EVAL_BERTSCORE:-0}" == "1" ]]; then
  eval_args+=(--bertscore --bertscore-model "${BERTSCORE_MODEL:-bert-base-multilingual-cased}" --bertscore-batch-size "${BERTSCORE_BATCH_SIZE:-16}")
  if [[ -n "${BERTSCORE_DEVICE:-}" ]]; then
    eval_args+=(--bertscore-device "$BERTSCORE_DEVICE")
  fi
fi
python3 scripts/evaluate_predictions.py "${eval_args[@]}"

echo "Predictions: $predictions"
echo "Metrics: $metrics"
