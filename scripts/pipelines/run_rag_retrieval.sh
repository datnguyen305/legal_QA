#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

RAG_METHOD=${RAG_METHOD:?RAG_METHOD is required}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/structured-single-hop-IR}
CONTEXT_DIR=${CONTEXT_DIR:-$REPO_ROOT/dataset/contexts}
STRUCTURED_DIR=${STRUCTURED_DIR:-$DATA_DIR/structured_data}
RAG_OUTPUT_DIR=${RAG_OUTPUT_DIR:-$REPO_ROOT/outputs/rag}
RAG_CORPUS_SCOPE=${RAG_CORPUS_SCOPE:-structured}
RAG_TOP_K=${RAG_TOP_K:-3}
RAG_LIMIT=${RAG_LIMIT:-}
RAG_CORPUS_LIMIT=${RAG_CORPUS_LIMIT:-}
RAG_PREDICTIONS=${RAG_PREDICTIONS:-1}
RAG_DENSE_COMPONENTS=${RAG_DENSE_COMPONENTS:-128}

limit_args=()
if [[ -n "$RAG_LIMIT" ]]; then
  limit_args+=(--limit "$RAG_LIMIT")
fi

corpus_limit_args=()
if [[ -n "$RAG_CORPUS_LIMIT" ]]; then
  corpus_limit_args+=(--corpus-limit "$RAG_CORPUS_LIMIT")
fi

prediction_args=()
if [[ "$RAG_PREDICTIONS" == "0" ]]; then
  prediction_args+=(--no-predictions)
fi

python3 "$REPO_ROOT/scripts/rag_retrieval.py" \
  --method "$RAG_METHOD" \
  --data-dir "$DATA_DIR" \
  --context-dir "$CONTEXT_DIR" \
  --structured-dir "$STRUCTURED_DIR" \
  --corpus-scope "$RAG_CORPUS_SCOPE" \
  --top-k "$RAG_TOP_K" \
  --dense-components "$RAG_DENSE_COMPONENTS" \
  --output-dir "$RAG_OUTPUT_DIR" \
  "${limit_args[@]}" \
  "${corpus_limit_args[@]}" \
  "${prediction_args[@]}"
