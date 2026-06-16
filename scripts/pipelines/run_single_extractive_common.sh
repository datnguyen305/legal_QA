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

EXTRACTIVE_BATCH_SIZE=${EXTRACTIVE_BATCH_SIZE:-16}
EXTRACTIVE_EPOCHS=${EXTRACTIVE_EPOCHS:-20}
EXTRACTIVE_PATIENCE=${EXTRACTIVE_PATIENCE:-3}
EXTRACTIVE_MAX_PASSAGES=${EXTRACTIVE_MAX_PASSAGES:-6}
EXTRACTIVE_PASSAGE_LEN=${EXTRACTIVE_PASSAGE_LEN:-256}
EXTRACTIVE_TOP_K=${EXTRACTIVE_TOP_K:-64}
EXTRACTIVE_CACHE_DIR=${EXTRACTIVE_CACHE_DIR:-cache/extractive}
EXTRACTIVE_DISK_CACHE=${EXTRACTIVE_DISK_CACHE:-1}
EXTRACTIVE_REBUILD_CACHE=${EXTRACTIVE_REBUILD_CACHE:-0}

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

cache_args=(--cache-dir "$EXTRACTIVE_CACHE_DIR")
if [[ "$EXTRACTIVE_DISK_CACHE" == "0" ]]; then
  cache_args+=(--no-disk-cache)
fi
if [[ "$EXTRACTIVE_REBUILD_CACHE" == "1" ]]; then
  cache_args+=(--rebuild-cache)
fi

python3 "$TRAIN_SCRIPT" \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$EXTRACTIVE_BATCH_SIZE" \
  --epochs "$EXTRACTIVE_EPOCHS" \
  --patience "$EXTRACTIVE_PATIENCE" \
  --max-passages "$EXTRACTIVE_MAX_PASSAGES" \
  --passage-len "$EXTRACTIVE_PASSAGE_LEN" \
  --top-k "$EXTRACTIVE_TOP_K" \
  "${cache_args[@]}" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 "$RUN_SCRIPT" \
  --model-dir "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output "$PREDICTIONS" \
  $(limit_args limit "$TEST_LIMIT")

python3 scripts/evaluate_predictions.py \
  --predictions "$PREDICTIONS" \
  --output "$METRICS" \
  "${bert_args[@]}"
