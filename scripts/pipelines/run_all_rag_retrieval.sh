#!/usr/bin/env bash
set -euo pipefail

METHODS=${RAG_METHODS:-ircot hipporag lightrag minirag raptor vi_hermes}

for method in $METHODS; do
  echo "===== Running RAG retrieval: $method ====="
  RAG_METHOD="$method" scripts/pipelines/run_rag_retrieval.sh
done
