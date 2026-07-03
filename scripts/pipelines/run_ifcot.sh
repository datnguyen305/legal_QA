#!/usr/bin/env bash
set -euo pipefail

RAG_METHOD=ifcot \
RAG_BM25_BACKEND=${RAG_BM25_BACKEND:-pyserini} \
RAG_PREDICTIONS=${RAG_PREDICTIONS:-0} \
scripts/pipelines/run_rag_retrieval.sh
