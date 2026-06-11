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
BERTSCORE=${BERTSCORE:-0}
BERTSCORE_MODEL=${BERTSCORE_MODEL:-bert-base-multilingual-cased}
AMP=${AMP:-bf16}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

CPG_BATCH_SIZE=${CPG_BATCH_SIZE:-16}
CPG_EPOCHS=${CPG_EPOCHS:-2}
CPG_MAX_CONTEXT_TOKENS=${CPG_MAX_CONTEXT_TOKENS:-1200}
CPG_MAX_ANSWER_TOKENS=${CPG_MAX_ANSWER_TOKENS:-96}

SNET_BATCH_SIZE=${SNET_BATCH_SIZE:-64}
SNET_EPOCHS=${SNET_EPOCHS:-2}
SNET_MAX_CONTEXT_TOKENS=${SNET_MAX_CONTEXT_TOKENS:-800}
SNET_MAX_ANSWER_TOKENS=${SNET_MAX_ANSWER_TOKENS:-96}

LATENTQA_BATCH_SIZE=${LATENTQA_BATCH_SIZE:-64}
LATENTQA_EPOCHS=${LATENTQA_EPOCHS:-2}
LATENTQA_MAX_CONTEXT_TOKENS=${LATENTQA_MAX_CONTEXT_TOKENS:-800}
LATENTQA_MAX_ANSWER_TOKENS=${LATENTQA_MAX_ANSWER_TOKENS:-96}

limit_args() {
  local name=$1
  local value=$2
  if [[ -n "$value" ]]; then
    printf -- "--%s %s" "$name" "$value"
  fi
}

bert_args=()
if [[ "$BERTSCORE" == "1" ]]; then
  bert_args=(--bertscore --bertscore-model "$BERTSCORE_MODEL")
fi

python3 scripts/train_cpg.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir models/cpg \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$CPG_BATCH_SIZE" \
  --epochs "$CPG_EPOCHS" \
  --max-context-tokens "$CPG_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$CPG_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_cpg.py --model-dir models/cpg --data "$TEST_DATA" --context-dir "$CONTEXT_DIR" --output outputs/cpg_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/cpg_predictions.jsonl --output outputs/cpg_metrics.json "${bert_args[@]}"

python3 scripts/train_snet.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir models/snet \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$SNET_BATCH_SIZE" \
  --epochs "$SNET_EPOCHS" \
  --max-context-tokens "$SNET_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$SNET_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_snet.py --model-dir models/snet --data "$TEST_DATA" --context-dir "$CONTEXT_DIR" --output outputs/snet_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/snet_predictions.jsonl --output outputs/snet_metrics.json "${bert_args[@]}"

python3 scripts/train_latentqa.py \
  --train-data "$TRAIN_DATA" \
  --dev-data "$DEV_DATA" \
  --context-dir "$CONTEXT_DIR" \
  --output-dir models/latentqa \
  --device "$DEVICE" \
  --amp "$AMP" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$LATENTQA_BATCH_SIZE" \
  --epochs "$LATENTQA_EPOCHS" \
  --max-context-tokens "$LATENTQA_MAX_CONTEXT_TOKENS" \
  --max-answer-tokens "$LATENTQA_MAX_ANSWER_TOKENS" \
  $(limit_args train-limit "$TRAIN_LIMIT") \
  $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_latentqa.py --model-dir models/latentqa --data "$TEST_DATA" --context-dir "$CONTEXT_DIR" --output outputs/latentqa_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/latentqa_predictions.jsonl --output outputs/latentqa_metrics.json "${bert_args[@]}"
