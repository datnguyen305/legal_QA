#!/usr/bin/env python3
"""Run TD-SAN / DynSAN inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.tdsan_preprocess import make_tdsan_record
from train_tdsan import encode, toks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/tdsan.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("TD-SAN inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.tdsan_model import TdsanForQuestionAnswering

    config = json.load(open(Path(args.model_dir) / "tdsan_config.json", encoding="utf-8"))
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TdsanForQuestionAnswering(
        vocab_size=config["vocab_size"],
        pad_token_id=config["pad_token_id"],
        hidden=config["hidden"],
        heads=config["heads"],
        top_k=config["top_k"],
        max_position=max(config["max_question_tokens"], config["max_passage_tokens"]) + 8,
    ).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()

    rows = []
    for idx, ex in enumerate(load_examples(args.data, args.limit), start=1):
        record = make_tdsan_record(ex, args.context_dir, args.max_context_chars)
        if record is None:
            prediction = ""
        else:
            passage = " ".join(record["passages"])
            passage_tokens = toks(passage)
            question_ids = torch.tensor(
                [encode(toks(record["question"]), vocab, config["max_question_tokens"])],
                dtype=torch.long,
                device=device,
            )
            passage_ids = torch.tensor(
                [encode(passage_tokens, vocab, config["max_passage_tokens"])],
                dtype=torch.long,
                device=device,
            )
            rank_ids = torch.zeros_like(passage_ids)
            with torch.no_grad():
                out = model(question_ids=question_ids, passage_ids=passage_ids, passage_rank_ids=rank_ids)
            start = int(out.start_logits[0].argmax().detach().cpu())
            end = int(out.end_logits[0, start:].argmax().detach().cpu()) + start
            prediction = " ".join(passage_tokens[start : end + 1]).strip()
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": "TD-SAN/DynSAN",
            }
        )
        if idx % 50 == 0:
            print(f"Processed {idx}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
