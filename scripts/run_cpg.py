#!/usr/bin/env python3
"""Run curriculum pointer-generator inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.cpg_preprocess import make_cpg_record
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import tokenize
from train_cpg import encode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/cpg.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Curriculum pointer-generator inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.cpg_model import CurriculumPointerGenerator

    config = json.load(open(Path(args.model_dir) / "cpg_config.json", encoding="utf-8"))
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    max_context_tokens = args.max_context_tokens or config["max_context_tokens"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CurriculumPointerGenerator(
        config["vocab_size"],
        vocab["<pad>"],
        vocab["<unk>"],
        vocab["<bos>"],
        vocab["<eos>"],
        hidden=config["hidden"],
        decoder_hidden=config["decoder_hidden"],
        block_size=config["block_size"],
    ).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()

    rows = []
    for ex in load_examples(args.data, args.limit):
        record = make_cpg_record(ex, args.context_dir, args.chunk_size, max_context_tokens, query_mode="question")
        prediction = ""
        if record is not None:
            context_tokens = tokenize(record["context"])
            batch = {
                "context_ids": torch.tensor(
                    [encode(context_tokens, vocab, max_context_tokens)],
                    dtype=torch.long,
                    device=device,
                ),
                "question_ids": torch.tensor(
                    [encode(tokenize(record["question"]), vocab, config["max_question_tokens"])],
                    dtype=torch.long,
                    device=device,
                ),
            }
            with torch.no_grad():
                out = model(**batch, max_answer_len=config["max_answer_tokens"])
            ids = out.logits[0].argmax(dim=-1).detach().cpu().tolist()
            words = []
            for idx in ids:
                word = inv_vocab.get(idx, "<unk>")
                if word == "<eos>":
                    break
                if word not in {"<pad>", "<bos>"}:
                    words.append(word)
            prediction = " ".join(words)
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": "IAL-CPG",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
