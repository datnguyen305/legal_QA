#!/usr/bin/env bash
set -euo pipefail
MODEL=dcmn_plus MODEL_DIR=${MODEL_DIR:-models/dcmn_plus} PREDICTIONS=${PREDICTIONS:-outputs/dcmn_plus_predictions.jsonl} METRICS=${METRICS:-outputs/dcmn_plus_metrics.json} scripts/pipelines/run_abstractive_baseline.sh
