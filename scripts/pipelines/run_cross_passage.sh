#!/usr/bin/env bash
set -euo pipefail

MODEL=cross_passage
TRAIN_SCRIPT=scripts/train_cross_passage.py
RUN_SCRIPT=scripts/run_cross_passage.py
MODEL_DIR=${MODEL_DIR:-models/cross_passage}
PREDICTIONS=${PREDICTIONS:-outputs/cross_passage_predictions.jsonl}
METRICS=${METRICS:-outputs/cross_passage_metrics.json}

source scripts/pipelines/run_single_extractive_common.sh
