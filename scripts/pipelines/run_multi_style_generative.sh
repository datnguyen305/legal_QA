#!/usr/bin/env bash
set -euo pipefail
MODEL=multi_style_generative MODEL_DIR=${MODEL_DIR:-models/multi_style_generative} PREDICTIONS=${PREDICTIONS:-outputs/multi_style_generative_predictions.jsonl} METRICS=${METRICS:-outputs/multi_style_generative_metrics.json} scripts/pipelines/run_abstractive_baseline.sh
