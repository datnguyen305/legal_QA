#!/usr/bin/env python3
"""Run inference for a fine-tuned pretrained seq2seq Legal QA model."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from pathlib import Path

from data_preprocessing.cpg_preprocess import progress_bar, sample_gold_context
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import normalize_space
from train_hf_seq2seq import make_input


def disable_apex_import() -> None:
    sys.modules["apex"] = None
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "apex" or name.startswith("apex."):
            raise ImportError("Apex import disabled by DISABLE_APEX=1")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        if os.environ.get("DISABLE_APEX", "1") == "1":
            disable_apex_import()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Pretrained seq2seq inference requires torch and transformers.") from exc

    config_path = Path(args.model_dir) / "seq2seq_config.json"
    config = json.load(open(config_path, encoding="utf-8")) if config_path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    examples = load_examples(args.data, args.limit)
    rows = []
    for start in range(0, len(examples), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        sources = []
        for ex in batch_examples:
            context = sample_gold_context(ex, args.context_dir)[: config.get("max_context_chars", 12000)]
            question = normalize_space(ex.get("question", ""))
            sources.append(make_input(question, context))
        encoded = tokenizer(
            sources,
            max_length=config.get("max_input_length", 1024),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": config.get("max_target_length", 256),
            "num_beams": args.num_beams or config.get("num_beams", 4),
        }
        forced_bos_token_id = config.get("forced_bos_token_id")
        if forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
        with torch.no_grad():
            ids = model.generate(**encoded, **generate_kwargs)
        predictions = tokenizer.batch_decode(ids, skip_special_tokens=True)
        for ex, prediction in zip(batch_examples, predictions):
            rows.append(
                {
                    "id": ex.get("id"),
                    "question": ex.get("question", ""),
                    "reference": ex.get("answer", ""),
                    "prediction": prediction,
                    "model": config.get("model_name", args.model_dir),
                }
            )
        progress_bar("Generate test seq2seq", min(start + args.batch_size, len(examples)), len(examples), len(rows))
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
