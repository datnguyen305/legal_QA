#!/usr/bin/env python3
"""Run inference for fine-tuned pretrained encoder extractive QA models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.cpg_preprocess import progress_bar
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import make_extractive_record
from train_hf_extractive import decode_best_span, extend_position_capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/QA/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for CLI consistency; inference decodes one sample at a time.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("HF extractive inference requires torch and transformers.") from exc

    config = json.load(open(Path(args.model_dir) / "hf_extractive_config.json", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    extend_position_capacity(model, tokenizer, config.get("max_length", 512))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    rows = []
    examples = load_examples(args.data, args.limit)
    for idx, ex in enumerate(examples, start=1):
        record = make_extractive_record(ex, args.context_dir, config.get("max_context_chars", 12000))
        if record is None:
            continue
        prediction = decode_best_span(
            record,
            tokenizer,
            model,
            device,
            config.get("max_length", 512),
            config.get("doc_stride", 128),
            config.get("max_answer_tokens", 160),
        )
        rows.append(
            {
                "id": record.get("id"),
                "question": record.get("question", ""),
                "reference": record.get("answer", ""),
                "abstractive_reference": record.get("reference", ex.get("answer", "")),
                "prediction": prediction,
                "model": config.get("model_name", args.model_dir),
            }
        )
        if idx == len(examples) or idx % 500 == 0:
            progress_bar("Generate HF extractive test", idx, len(examples), len(rows))
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
