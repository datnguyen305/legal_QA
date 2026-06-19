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

MODEL_KEY=${MODEL_KEY:-mbert}
MODEL_NAME=${MODEL_NAME:-}
MODEL_DIR=${MODEL_DIR:-}
PREDICTIONS=${PREDICTIONS:-}
METRICS=${METRICS:-}

HF_EXTRACTIVE_BATCH_SIZE=${HF_EXTRACTIVE_BATCH_SIZE:-16}
HF_EXTRACTIVE_EPOCHS=${HF_EXTRACTIVE_EPOCHS:-5}
HF_EXTRACTIVE_PATIENCE=${HF_EXTRACTIVE_PATIENCE:-2}
HF_EXTRACTIVE_LR=${HF_EXTRACTIVE_LR:-3e-5}
HF_EXTRACTIVE_MAX_LENGTH=${HF_EXTRACTIVE_MAX_LENGTH:-}
HF_EXTRACTIVE_DOC_STRIDE=${HF_EXTRACTIVE_DOC_STRIDE:-128}
HF_EXTRACTIVE_MAX_ANSWER_TOKENS=${HF_EXTRACTIVE_MAX_ANSWER_TOKENS:-160}
HF_EXTRACTIVE_MAX_CONTEXT_CHARS=${HF_EXTRACTIVE_MAX_CONTEXT_CHARS:-12000}

case "$MODEL_KEY" in
  legal_bert)
    DEFAULT_MODEL_NAME="nlpaueb/legal-bert-base-uncased"
    DEFAULT_MODEL_DIR="models/legal_bert_extractive"
    DEFAULT_PREDICTIONS="outputs/legal_bert_predictions.jsonl"
    DEFAULT_METRICS="outputs/legal_bert_metrics.json"
    ;;
  videberta)
    DEFAULT_MODEL_NAME="Fsoft-AIC/videberta-base"
    DEFAULT_MODEL_DIR="models/videberta_extractive"
    DEFAULT_PREDICTIONS="outputs/videberta_predictions.jsonl"
    DEFAULT_METRICS="outputs/videberta_metrics.json"
    ;;
  mbert)
    DEFAULT_MODEL_NAME="bert-base-multilingual-cased"
    DEFAULT_MODEL_DIR="models/mbert_extractive"
    DEFAULT_PREDICTIONS="outputs/mbert_predictions.jsonl"
    DEFAULT_METRICS="outputs/mbert_metrics.json"
    ;;
  xlmr|xlm_roberta)
    DEFAULT_MODEL_NAME="xlm-roberta-base"
    DEFAULT_MODEL_DIR="models/xlmr_extractive"
    DEFAULT_PREDICTIONS="outputs/xlmr_predictions.jsonl"
    DEFAULT_METRICS="outputs/xlmr_metrics.json"
    ;;
  phobert)
    DEFAULT_MODEL_NAME="vinai/phobert-base"
    DEFAULT_MODEL_DIR="models/phobert_extractive"
    DEFAULT_PREDICTIONS="outputs/phobert_predictions.jsonl"
    DEFAULT_METRICS="outputs/phobert_metrics.json"
    ;;
  *)
    DEFAULT_MODEL_NAME="$MODEL_KEY"
    DEFAULT_MODEL_DIR="models/${MODEL_KEY//\//_}_extractive"
    DEFAULT_PREDICTIONS="outputs/${MODEL_KEY//\//_}_extractive_predictions.jsonl"
    DEFAULT_METRICS="outputs/${MODEL_KEY//\//_}_extractive_metrics.json"
    ;;
esac

MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
PREDICTIONS=${PREDICTIONS:-$DEFAULT_PREDICTIONS}
METRICS=${METRICS:-$DEFAULT_METRICS}
HF_EXTRACTIVE_MAX_LENGTH=${HF_EXTRACTIVE_MAX_LENGTH:-512}

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

python3 scripts/train_hf_extractive.py \
  --model-name "$MODEL_NAME" \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$HF_EXTRACTIVE_BATCH_SIZE" \
  --epochs "$HF_EXTRACTIVE_EPOCHS" \
  --patience "$HF_EXTRACTIVE_PATIENCE" \
  --lr "$HF_EXTRACTIVE_LR" \
  --max-context-chars "$HF_EXTRACTIVE_MAX_CONTEXT_CHARS" \
  --max-length "$HF_EXTRACTIVE_MAX_LENGTH" \
  --doc-stride "$HF_EXTRACTIVE_DOC_STRIDE" \
  --max-answer-tokens "$HF_EXTRACTIVE_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")

python3 scripts/run_hf_extractive.py \
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
