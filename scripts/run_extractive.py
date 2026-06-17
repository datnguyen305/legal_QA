#!/usr/bin/env python3
"""Run inference for extractive paper models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.cpg_preprocess import sample_gold_context
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import make_extractive_record, normalize_space, tokenize
from train_cpg import encode
from train_extractive import MODEL_CHOICES, make_model, split_passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-model", choices=MODEL_CHOICES, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Extractive inference requires PyTorch.") from exc

    config = json.load(open(Path(args.model_dir) / "extractive_config.json", encoding="utf-8"))
    model_name = config["model"]
    if model_name not in MODEL_CHOICES:
        raise SystemExit(f"Unsupported extractive model: {model_name}")
    if args.expected_model is not None and model_name != args.expected_model:
        raise SystemExit(f"{args.model_dir} contains model={model_name}, expected {args.expected_model}.")
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, config, vocab, device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()
    rows = []
    for ex in load_examples(args.data, args.limit):
        question = normalize_space(ex.get("question", ""))
        record = make_extractive_record(ex, args.context_dir, config["max_context_chars"])
        context = record["context"] if record is not None else sample_gold_context(ex, args.context_dir)[: config["max_context_chars"]]
        context_tokens = tokenize(context)
        extractive_reference = record["answer"] if record is not None else ex.get("answer", "")
        passages = split_passages(context_tokens, config["max_passages"], config["passage_len"])
        while len(passages) < config["max_passages"]:
            passages.append([])
        passage_ids = [encode(p, vocab, config["passage_len"]) for p in passages[: config["max_passages"]]]
        batch = {
            "passage_ids": torch.tensor([passage_ids], dtype=torch.long, device=device),
            "context_ids": torch.tensor([[tok for p in passage_ids for tok in p]], dtype=torch.long, device=device),
            "question_ids": torch.tensor([encode(tokenize(question), vocab, config["max_question_tokens"])], dtype=torch.long, device=device),
        }
        if model_name == "qanet":
            batch.pop("passage_ids")
        with torch.no_grad():
            out = model(**batch)
        start = int(out.start_logits.argmax(dim=-1).item())
        end = int(out.end_logits.argmax(dim=-1).item())
        if end < start:
            end = start
        flat_tokens = [tok for passage in passages for tok in passage]
        end = min(end, len(flat_tokens) - 1)
        rows.append(
            {
                "id": ex.get("id"),
                "question": question,
                "reference": extractive_reference,
                "abstractive_reference": ex.get("answer", ""),
                "prediction": " ".join(flat_tokens[start : end + 1]),
                "model": model_name,
            }
        )
    output = args.output or f"outputs/{model_name}_predictions.jsonl"
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
