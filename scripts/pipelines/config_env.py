#!/usr/bin/env python3
"""Emit bash defaults from a model JSON config.

The generated shell code only assigns variables that are not already set, so
explicit environment overrides keep working:

    EPOCHS=5 bash scripts/pipelines/run_equals_pipeline.sh
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any


KEY_MAP = {
    "data.train_data": "TRAIN_DATA",
    "data.dev_data": "DEV_DATA",
    "data.test_data": "TEST_DATA",
    "data.context_dir": "CONTEXT_DIR",
    "paths.model_dir": "MODEL_DIR",
    "paths.output_dir": "OUTPUT_DIR",
    "paths.predictions": "PREDICTIONS",
    "paths.metrics": "METRICS",
    "train.base_model": "BASE_MODEL",
    "train.batch_size": "BATCH_SIZE",
    "train.epochs": "EPOCHS",
    "train.selector_epochs": "SELECTOR_EPOCHS",
    "train.answer_epochs": "ANSWER_EPOCHS",
    "train.ranker_epochs": "RANKER_EPOCHS",
    "train.learning_rate": "LR",
    "train.max_context_chars": "MAX_CONTEXT_CHARS",
    "train.max_length": "MAX_LENGTH",
    "train.max_sentences": "MAX_SENTENCES",
    "train.max_docs": "MAX_DOCS",
    "train.max_passages": "MAX_PASSAGES",
    "train.max_paragraphs": "MAX_PARAGRAPHS",
    "train.max_question_tokens": "MAX_QUESTION_TOKENS",
    "train.max_passage_tokens": "MAX_PASSAGE_TOKENS",
    "train.max_context_tokens": "MAX_CONTEXT_TOKENS",
    "train.max_answer_tokens": "MAX_ANSWER_TOKENS",
    "train.hidden": "HIDDEN",
    "train.heads": "HEADS",
    "train.top_k": "TOP_K",
    "train.chunk_sizes": "CHUNK_SIZES",
    "train.easy_ratio": "EASY_RATIO",
    "train.easy_ratio_decay": "EASY_RATIO_DECAY",
    "train.decoder_hidden": "DECODER_HIDDEN",
    "train.block_size": "BLOCK_SIZE",
    "train.window_tokens": "WINDOW_TOKENS",
    "train.stride": "STRIDE",
    "train.early_layer": "EARLY_LAYER",
    "train.max_candidates": "MAX_CANDIDATES",
    "train.train_limit": "TRAIN_LIMIT",
    "train.dev_limit": "DEV_LIMIT",
    "train.device": "DEVICE",
    "train.skip_train": "SKIP_TRAIN",
    "train.num_workers": "NUM_WORKERS",
    "train.preprocess_num_proc": "PREPROCESS_NUM_PROC",
    "predict.retriever": "RETRIEVER",
    "predict.sbert_model": "SBERT_MODEL",
    "predict.top_k": "TOP_K",
    "predict.max_answer_length": "MAX_ANSWER_LENGTH",
    "predict.chunk_size": "CHUNK_SIZE",
    "predict.limit": "LIMIT",
    "evaluation.upper_bound": "EVAL_UPPER_BOUND",
    "evaluation.bertscore": "EVAL_BERTSCORE",
    "evaluation.bertscore_model": "BERTSCORE_MODEL",
    "evaluation.bertscore_batch_size": "BERTSCORE_BATCH_SIZE",
    "evaluation.bertscore_device": "BERTSCORE_DEVICE",
}


def get_path(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def shell_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def emit_default(name: str, value: str) -> str:
    return f'if [[ -z "${{{name}+x}}" ]]; then {name}={shlex.quote(value)}; fi'


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: config_env.py CONFIG_JSON")
    config_path = Path(sys.argv[1])
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"CONFIG={shlex.quote(str(config_path))}")
    for dotted, env_name in KEY_MAP.items():
        raw_value = get_path(config, dotted)
        value = shell_value(raw_value)
        if value is not None:
            print(emit_default(env_name, value))


if __name__ == "__main__":
    main()
