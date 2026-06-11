#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-dataset}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/train_data.json}
DEV_DATA=${DEV_DATA:-$DATA_DIR/dev_data.json}
TEST_DATA=${TEST_DATA:-$DATA_DIR/test_data.json}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
DEV_LIMIT=${DEV_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
BERTSCORE=${BERTSCORE:-0}
BERTSCORE_MODEL=${BERTSCORE_MODEL:-bert-base-multilingual-cased}

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

python3 scripts/train_cpg.py --train-data "$TRAIN_DATA" --dev-data "$DEV_DATA" --output-dir models/cpg $(limit_args train-limit "$TRAIN_LIMIT") $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_cpg.py --model-dir models/cpg --data "$TEST_DATA" --output outputs/cpg_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/cpg_predictions.jsonl --output outputs/cpg_metrics.json "${bert_args[@]}"

python3 scripts/train_snet.py --train-data "$TRAIN_DATA" --dev-data "$DEV_DATA" --output-dir models/snet $(limit_args train-limit "$TRAIN_LIMIT") $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_snet.py --model-dir models/snet --data "$TEST_DATA" --output outputs/snet_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/snet_predictions.jsonl --output outputs/snet_metrics.json "${bert_args[@]}"

python3 scripts/train_latentqa.py --train-data "$TRAIN_DATA" --dev-data "$DEV_DATA" --output-dir models/latentqa $(limit_args train-limit "$TRAIN_LIMIT") $(limit_args dev-limit "$DEV_LIMIT")
python3 scripts/run_latentqa.py --model-dir models/latentqa --data "$TEST_DATA" --output outputs/latentqa_predictions.jsonl $(limit_args limit "$TEST_LIMIT")
python3 scripts/evaluate_predictions.py --predictions outputs/latentqa_predictions.jsonl --output outputs/latentqa_metrics.json "${bert_args[@]}"
