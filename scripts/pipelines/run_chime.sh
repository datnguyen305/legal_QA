#!/usr/bin/env bash
set -euo pipefail
MODEL=chime MODEL_DIR=${MODEL_DIR:-models/chime} PREDICTIONS=${PREDICTIONS:-outputs/chime_predictions.jsonl} METRICS=${METRICS:-outputs/chime_metrics.json} scripts/pipelines/run_abstractive_baseline.sh
