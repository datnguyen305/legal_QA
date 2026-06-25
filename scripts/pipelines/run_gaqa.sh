#!/usr/bin/env bash
set -euo pipefail
MODEL=gaqa MODEL_DIR=${MODEL_DIR:-models/gaqa} PREDICTIONS=${PREDICTIONS:-outputs/gaqa_predictions.jsonl} METRICS=${METRICS:-outputs/gaqa_metrics.json} scripts/pipelines/run_abstractive_baseline.sh
