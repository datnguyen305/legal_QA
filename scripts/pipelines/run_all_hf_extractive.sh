#!/usr/bin/env bash
set -euo pipefail

MODELS=${HF_EXTRACTIVE_MODELS:-legal_bert videberta mbert xlmr phobert}

for model in $MODELS; do
  echo "===== Running HF extractive model: $model ====="
  MODEL_KEY="$model" scripts/pipelines/run_hf_extractive.sh
done
