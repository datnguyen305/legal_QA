#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/structured-single-hop-IR}
CONTEXT_DIR=${CONTEXT_DIR:-$REPO_ROOT/dataset/contexts}
STRUCTURED_DIR=${STRUCTURED_DIR:-$DATA_DIR/structured_data}
BM25_OUTPUT_DIR=${BM25_OUTPUT_DIR:-$REPO_ROOT/outputs/bm25}
BM25_TOP_K=${BM25_TOP_K:-3}
BM25_CORPUS_SCOPE=${BM25_CORPUS_SCOPE:-structured}
BM25_RETRIEVER=${BM25_RETRIEVER:-hybrid}
BM25_HYBRID_WEIGHT=${BM25_HYBRID_WEIGHT:-0.5}
BM25_HYBRID_CANDIDATES=${BM25_HYBRID_CANDIDATES:-100}
BM25_DENSE_COMPONENTS=${BM25_DENSE_COMPONENTS:-256}
BM25_LIMIT=${BM25_LIMIT:-}
BM25_CORPUS_LIMIT=${BM25_CORPUS_LIMIT:-}
BM25_PREDICTIONS=${BM25_PREDICTIONS:-1}
BM25_QUIET=${BM25_QUIET:-0}

limit_args=()
if [[ -n "$BM25_LIMIT" ]]; then
  limit_args+=(--limit "$BM25_LIMIT")
fi

prediction_args=()
if [[ "$BM25_PREDICTIONS" == "0" ]]; then
  prediction_args+=(--no-predictions)
fi
if [[ "$BM25_QUIET" == "1" ]]; then
  prediction_args+=(--quiet)
fi

corpus_limit_args=()
if [[ -n "$BM25_CORPUS_LIMIT" ]]; then
  corpus_limit_args+=(--corpus-limit "$BM25_CORPUS_LIMIT")
fi

python3 "$REPO_ROOT/scripts/bm25_retrieval.py" \
  --data-dir "$DATA_DIR" \
  --context-dir "$CONTEXT_DIR" \
  --structured-dir "$STRUCTURED_DIR" \
  --corpus-scope "$BM25_CORPUS_SCOPE" \
  --retriever "$BM25_RETRIEVER" \
  --top-k "$BM25_TOP_K" \
  --hybrid-bm25-weight "$BM25_HYBRID_WEIGHT" \
  --hybrid-candidates "$BM25_HYBRID_CANDIDATES" \
  --dense-components "$BM25_DENSE_COMPONENTS" \
  --output-dir "$BM25_OUTPUT_DIR" \
  "${limit_args[@]}" \
  "${corpus_limit_args[@]}" \
  "${prediction_args[@]}"
