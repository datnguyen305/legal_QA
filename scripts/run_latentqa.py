#!/usr/bin/env python3
"""Run LatentQA inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.cpg_preprocess import sample_gold_context
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from train_cpg import encode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/latentqa_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("LatentQA inference requires PyTorch.") from exc

    from model_architectures.latentqa_model import LatentQA

    config = json.load(open(Path(args.model_dir) / "latentqa_config.json", encoding="utf-8"))
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentQA(config["vocab_size"], vocab["<pad>"], vocab["<bos>"], vocab["<eos>"], config["hidden"], config["decoder_hidden"]).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()
    rows = []
    for ex in load_examples(args.data, args.limit):
        context = sample_gold_context(ex, args.context_dir)[: config["max_context_chars"]]
        batch = {
            "context_ids": torch.tensor([encode(tokenize(context), vocab, config["max_context_tokens"])], dtype=torch.long, device=device),
            "question_ids": torch.tensor([encode(tokenize(ex.get("question", "")), vocab, config["max_question_tokens"])], dtype=torch.long, device=device),
        }
        with torch.no_grad():
            out = model(**batch, max_answer_len=config["max_answer_tokens"])
        words = []
        for idx in out.logits[0].argmax(dim=-1).detach().cpu().tolist():
            word = inv_vocab.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in {"<pad>", "<bos>"}:
                words.append(word)
        rows.append({"id": ex.get("id"), "question": ex.get("question", ""), "reference": ex.get("answer", ""), "prediction": " ".join(words), "model": "LatentQA"})
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
