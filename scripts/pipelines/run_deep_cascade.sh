#!/usr/bin/env bash
set -euo pipefail

MODEL=deep_cascade
TRAIN_SCRIPT=scripts/train_deep_cascade.py
RUN_SCRIPT=scripts/run_deep_cascade.py
MODEL_DIR=${MODEL_DIR:-models/deep_cascade}
PREDICTIONS=${PREDICTIONS:-outputs/deep_cascade_predictions.jsonl}
METRICS=${METRICS:-outputs/deep_cascade_metrics.json}

source scripts/pipelines/run_single_extractive_common.sh
