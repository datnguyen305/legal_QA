#!/usr/bin/env bash
set -euo pipefail

MODELS=${EXTRACTIVE_MODELS:-qanet cross_passage deep_cascade td_san}

for model in $MODELS; do
  echo "===== Running extractive model: $model ====="
  case "$model" in
    qanet)
      scripts/pipelines/run_qanet.sh
      ;;
    cross_passage)
      scripts/pipelines/run_cross_passage.sh
      ;;
    deep_cascade)
      scripts/pipelines/run_deep_cascade.sh
      ;;
    td_san)
      scripts/pipelines/run_td_san.sh
      ;;
    *)
      echo "Unknown extractive model: $model" >&2
      exit 1
      ;;
  esac
done
