#!/usr/bin/env bash
set -euo pipefail

MODEL=qanet
TRAIN_SCRIPT=scripts/train_qanet.py
RUN_SCRIPT=scripts/run_qanet.py
MODEL_DIR=${MODEL_DIR:-models/qanet}
PREDICTIONS=${PREDICTIONS:-outputs/qanet_predictions.jsonl}
METRICS=${METRICS:-outputs/qanet_metrics.json}

source scripts/pipelines/run_single_extractive_common.sh
