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
PIN_MEMORY=${PIN_MEMORY:-0}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-0}
DEVICE=${DEVICE:-cuda}
BERTSCORE_DEVICE=${BERTSCORE_DEVICE:-$DEVICE}
export DISABLE_APEX=${DISABLE_APEX:-1}
export PYTORCH_NVML_BASED_CUDA_CHECK=${PYTORCH_NVML_BASED_CUDA_CHECK:-0}

MODEL_KEY=${MODEL_KEY:-vit5}
MODEL_NAME=${MODEL_NAME:-}
MODEL_DIR=${MODEL_DIR:-}
PREDICTIONS=${PREDICTIONS:-}
METRICS=${METRICS:-}

SEQ2SEQ_BATCH_SIZE=${SEQ2SEQ_BATCH_SIZE:-8}
SEQ2SEQ_GRAD_ACCUM_STEPS=${SEQ2SEQ_GRAD_ACCUM_STEPS:-2}
SEQ2SEQ_GRADIENT_CHECKPOINTING=${SEQ2SEQ_GRADIENT_CHECKPOINTING:-0}
SEQ2SEQ_EPOCHS=${SEQ2SEQ_EPOCHS:-10}
SEQ2SEQ_PATIENCE=${SEQ2SEQ_PATIENCE:-3}
SEQ2SEQ_LR=${SEQ2SEQ_LR:-3e-5}
SEQ2SEQ_MAX_INPUT_LENGTH=${SEQ2SEQ_MAX_INPUT_LENGTH:-1024}
SEQ2SEQ_MAX_TARGET_LENGTH=${SEQ2SEQ_MAX_TARGET_LENGTH:-256}
SEQ2SEQ_NUM_BEAMS=${SEQ2SEQ_NUM_BEAMS:-1}
SEQ2SEQ_DEV_NUM_BEAMS=${SEQ2SEQ_DEV_NUM_BEAMS:-1}
SEQ2SEQ_DEV_EVAL_LIMIT=${SEQ2SEQ_DEV_EVAL_LIMIT:-2000}
SEQ2SEQ_TEST_BATCH_SIZE=${SEQ2SEQ_TEST_BATCH_SIZE:-8}
SEQ2SEQ_DEV_GENERATE_BATCH_SIZE=${SEQ2SEQ_DEV_GENERATE_BATCH_SIZE:-64}
SEQ2SEQ_SRC_LANG=${SEQ2SEQ_SRC_LANG:-}
SEQ2SEQ_TGT_LANG=${SEQ2SEQ_TGT_LANG:-}

case "$MODEL_KEY" in
  vit5)
    DEFAULT_MODEL_NAME="VietAI/vit5-base"
    DEFAULT_MODEL_DIR="models/vit5"
    DEFAULT_PREDICTIONS="outputs/vit5_predictions.jsonl"
    DEFAULT_METRICS="outputs/vit5_metrics.json"
    ;;
  bartpho)
    DEFAULT_MODEL_NAME="vinai/bartpho-syllable"
    DEFAULT_MODEL_DIR="models/bartpho"
    DEFAULT_PREDICTIONS="outputs/bartpho_predictions.jsonl"
    DEFAULT_METRICS="outputs/bartpho_metrics.json"
    ;;
  mt5)
    DEFAULT_MODEL_NAME="google/mt5-base"
    DEFAULT_MODEL_DIR="models/mt5"
    DEFAULT_PREDICTIONS="outputs/mt5_predictions.jsonl"
    DEFAULT_METRICS="outputs/mt5_metrics.json"
    ;;
  mbart)
    DEFAULT_MODEL_NAME="facebook/mbart-large-50-many-to-many-mmt"
    DEFAULT_MODEL_DIR="models/mbart"
    DEFAULT_PREDICTIONS="outputs/mbart_predictions.jsonl"
    DEFAULT_METRICS="outputs/mbart_metrics.json"
    SEQ2SEQ_SRC_LANG=${SEQ2SEQ_SRC_LANG:-vi_VN}
    SEQ2SEQ_TGT_LANG=${SEQ2SEQ_TGT_LANG:-vi_VN}
    ;;
  *)
    DEFAULT_MODEL_NAME="$MODEL_KEY"
    DEFAULT_MODEL_DIR="models/${MODEL_KEY//\//_}"
    DEFAULT_PREDICTIONS="outputs/${MODEL_KEY//\//_}_predictions.jsonl"
    DEFAULT_METRICS="outputs/${MODEL_KEY//\//_}_metrics.json"
    ;;
esac

MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
PREDICTIONS=${PREDICTIONS:-$DEFAULT_PREDICTIONS}
METRICS=${METRICS:-$DEFAULT_METRICS}

limit_args() {
  local name=$1
  local value=$2
  if [[ -n "$value" ]]; then
    printf -- "--%s %s" "$name" "$value"
  fi
}

lang_args=()
if [[ -n "$SEQ2SEQ_SRC_LANG" ]]; then
  lang_args+=(--src-lang "$SEQ2SEQ_SRC_LANG")
fi
if [[ -n "$SEQ2SEQ_TGT_LANG" ]]; then
  lang_args+=(--tgt-lang "$SEQ2SEQ_TGT_LANG")
fi

dev_gen_args=()
if [[ -n "$SEQ2SEQ_DEV_GENERATE_BATCH_SIZE" ]]; then
  dev_gen_args+=(--dev-generate-batch-size "$SEQ2SEQ_DEV_GENERATE_BATCH_SIZE")
fi

bert_args=(--bertscore-model "$BERTSCORE_MODEL" --bertscore-batch-size "$BERTSCORE_BATCH_SIZE")
if [[ -n "$BERTSCORE_DEVICE" ]]; then
  bert_args+=(--bertscore-device "$BERTSCORE_DEVICE")
fi
if [[ "$BERTSCORE" == "0" ]]; then
  bert_args=(--no-bertscore)
fi

python3 scripts/train_hf_seq2seq.py \
  --model-name "$MODEL_NAME" \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory "$PIN_MEMORY" \
  --persistent-workers "$PERSISTENT_WORKERS" \
  --batch-size "$SEQ2SEQ_BATCH_SIZE" \
  --grad-accum-steps "$SEQ2SEQ_GRAD_ACCUM_STEPS" \
  --gradient-checkpointing "$SEQ2SEQ_GRADIENT_CHECKPOINTING" \
  --epochs "$SEQ2SEQ_EPOCHS" \
  --patience "$SEQ2SEQ_PATIENCE" \
  --lr "$SEQ2SEQ_LR" \
  --max-input-length "$SEQ2SEQ_MAX_INPUT_LENGTH" \
  --max-target-length "$SEQ2SEQ_MAX_TARGET_LENGTH" \
  --num-beams "$SEQ2SEQ_NUM_BEAMS" \
  --dev-num-beams "$SEQ2SEQ_DEV_NUM_BEAMS" \
  --dev-eval-limit "$SEQ2SEQ_DEV_EVAL_LIMIT" \
  "${lang_args[@]}" \
  "${dev_gen_args[@]}" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 scripts/run_hf_seq2seq.py \
  --model-dir "$MODEL_DIR" \
  --data "$TEST_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output "$PREDICTIONS" \
  --batch-size "$SEQ2SEQ_TEST_BATCH_SIZE" \
  --device "$DEVICE" \
  $(limit_args limit "$TEST_LIMIT")

python3 scripts/evaluate_predictions.py \
  --predictions "$PREDICTIONS" \
  --output "$METRICS" \
  "${bert_args[@]}"
