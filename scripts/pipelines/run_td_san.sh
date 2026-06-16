#!/usr/bin/env bash
set -euo pipefail

MODEL=td_san
TRAIN_SCRIPT=scripts/train_td_san.py
RUN_SCRIPT=scripts/run_td_san.py
MODEL_DIR=${MODEL_DIR:-models/td_san}
PREDICTIONS=${PREDICTIONS:-outputs/td_san_predictions.jsonl}
METRICS=${METRICS:-outputs/td_san_metrics.json}

source scripts/pipelines/run_single_extractive_common.sh
