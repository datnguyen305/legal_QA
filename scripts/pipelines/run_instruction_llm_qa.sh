#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/QA}
TEST_DATA=${TEST_DATA:-$DATA_DIR/test_data.json}
CONTEXT_DIR=${CONTEXT_DIR:-$REPO_ROOT/dataset/contexts}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/instruction_llm}

TEST_LIMIT=${TEST_LIMIT:-}
LLM_BATCH_SIZE=${LLM_BATCH_SIZE:-1}
LLM_DEVICE=${LLM_DEVICE:-auto}
LLM_DTYPE=${LLM_DTYPE:-auto}
LLM_MAX_CONTEXT_CHARS=${LLM_MAX_CONTEXT_CHARS:-12000}
LLM_MAX_INPUT_TOKENS=${LLM_MAX_INPUT_TOKENS:-4096}
LLM_MAX_NEW_TOKENS=${LLM_MAX_NEW_TOKENS:-256}
LLM_TEMPERATURE=${LLM_TEMPERATURE:-0.0}
LLM_TOP_P=${LLM_TOP_P:-1.0}
LLM_LOAD_IN_4BIT=${LLM_LOAD_IN_4BIT:-0}
LLM_TRUST_REMOTE_CODE=${LLM_TRUST_REMOTE_CODE:-1}

BERTSCORE=${BERTSCORE:-1}
BERTSCORE_MODEL=${BERTSCORE_MODEL:-bert-base-multilingual-cased}
BERTSCORE_BATCH_SIZE=${BERTSCORE_BATCH_SIZE:-16}
BERTSCORE_DEVICE=${BERTSCORE_DEVICE:-cuda}

MODELS=${MODELS:-"llama31_8b qwen25_7b llama3_8b qwen25_14b gemma2_9b mistral7b_v03 phi4_mini glm4_9b"}

mkdir -p "$OUTPUT_DIR"

limit_args=()
if [[ -n "$TEST_LIMIT" ]]; then
  limit_args+=(--limit "$TEST_LIMIT")
fi

quant_args=()
if [[ "$LLM_LOAD_IN_4BIT" == "1" ]]; then
  quant_args+=(--load-in-4bit)
fi
if [[ "$LLM_TRUST_REMOTE_CODE" == "1" ]]; then
  quant_args+=(--trust-remote-code)
fi

bert_args=(--bertscore-model "$BERTSCORE_MODEL" --bertscore-batch-size "$BERTSCORE_BATCH_SIZE")
if [[ -n "$BERTSCORE_DEVICE" ]]; then
  bert_args+=(--bertscore-device "$BERTSCORE_DEVICE")
fi
if [[ "$BERTSCORE" == "0" ]]; then
  bert_args=(--no-bertscore)
fi

for model in $MODELS; do
  predictions="$OUTPUT_DIR/${model}_test_predictions.jsonl"
  metrics="$OUTPUT_DIR/${model}_test_metrics.json"
  echo "Running $model on QA test split"
  python3 scripts/run_instruction_llm_qa.py \
    --model "$model" \
    --data "$TEST_DATA" \
    --context-dir "$CONTEXT_DIR" \
    --output "$predictions" \
    --batch-size "$LLM_BATCH_SIZE" \
    --device "$LLM_DEVICE" \
    --dtype "$LLM_DTYPE" \
    --max-context-chars "$LLM_MAX_CONTEXT_CHARS" \
    --max-input-tokens "$LLM_MAX_INPUT_TOKENS" \
    --max-new-tokens "$LLM_MAX_NEW_TOKENS" \
    --temperature "$LLM_TEMPERATURE" \
    --top-p "$LLM_TOP_P" \
    "${limit_args[@]}" \
    "${quant_args[@]}"

  python3 scripts/evaluate_predictions.py \
    --predictions "$predictions" \
    --output "$metrics" \
    "${bert_args[@]}"
done
