#!/usr/bin/env python3
"""Run Deep Cascade Model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.deep_cascade_preprocess import load_deep_cascade_records
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from train_deep_cascade import encode, toks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/deep_cascade.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Deep Cascade inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.deep_cascade_model import DeepCascadeReader

    config = json.load(open(Path(args.model_dir) / "deep_cascade_config.json", encoding="utf-8"))
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reader = DeepCascadeReader(config["vocab_size"], config["pad_token_id"], hidden=config["hidden"]).to(device)
    reader.load_state_dict(torch.load(Path(args.model_dir) / "reader.bin", map_location=device))
    reader.eval()

    records = load_deep_cascade_records(
        args.data,
        args.context_dir,
        args.limit,
        config["max_docs"],
        config["max_paragraphs"],
        args.max_context_chars,
    )
    by_id = {row["id"]: row for row in records}
    rows = []
    for ex in load_examples(args.data, args.limit):
        row = by_id.get(ex.get("id"))
        prediction = ""
        if row is not None:
            context_tokens = toks(row["context"])
            question_tokens = toks(row["question"])
            token_features = []
            qset = set(question_tokens)
            for token in context_tokens[: config["max_context_tokens"]]:
                token_features.append([1.0 if token in qset else 0.0, 1.0 if token in {".", "!", "?", ";"} else 0.0, 0.0, 0.0, 0.0])
            token_features += [[0.0] * 5 for _ in range(config["max_context_tokens"] - len(token_features))]
            doc_spans = [[0, min(len(context_tokens), config["max_context_tokens"]) - 1]] + [[-1, -1] for _ in range(config["max_docs"] - 1)]
            para_spans = [[[0, min(len(context_tokens), config["max_context_tokens"]) - 1]] + [[-1, -1] for _ in range(config["max_paragraphs"] - 1)]]
            para_spans += [[[-1, -1] for _ in range(config["max_paragraphs"])] for _ in range(config["max_docs"] - 1)]
            batch = {
                "question_ids": torch.tensor([encode(question_tokens, vocab, config["max_question_tokens"])], dtype=torch.long, device=device),
                "context_ids": torch.tensor([encode(context_tokens, vocab, config["max_context_tokens"])], dtype=torch.long, device=device),
                "token_features": torch.tensor([token_features], dtype=torch.float, device=device),
                "doc_spans": torch.tensor([doc_spans], dtype=torch.long, device=device),
                "para_spans": torch.tensor([para_spans], dtype=torch.long, device=device),
                "doc_features": torch.tensor([row["doc_features"]], dtype=torch.float, device=device),
                "para_features": torch.tensor([row["para_features"]], dtype=torch.float, device=device),
            }
            with torch.no_grad():
                out = reader(**batch)
            start = int(out.start_logits[0].argmax().detach().cpu())
            end = int(out.end_logits[0, start:].argmax().detach().cpu()) + start
            prediction = " ".join(context_tokens[start : end + 1])
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": "Deep Cascade",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
