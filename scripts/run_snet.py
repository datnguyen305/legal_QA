#!/usr/bin/env python3
"""Run S-NET GRU extraction-then-synthesis inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.cpg_preprocess import sample_gold_context
from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from model_architectures.snet_model import SNetSynthesis, select_evidence_sentence, token_feature_flags
from train_cpg import encode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/snet_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("S-NET inference requires PyTorch.") from exc

    config = json.load(open(Path(args.model_dir) / "snet_config.json", encoding="utf-8"))
    vocab = json.load(open(Path(args.model_dir) / "vocab.json", encoding="utf-8"))
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SNetSynthesis(
        config["vocab_size"],
        vocab["<pad>"],
        vocab["<bos>"],
        vocab["<eos>"],
        embed_size=config["embed_size"],
        feature_size=config["feature_size"],
        hidden_size=config["hidden_size"],
    ).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()

    rows = []
    for ex in load_examples(args.data, args.limit):
        context = sample_gold_context(ex, args.context_dir)[: config["max_context_chars"]]
        question = normalize_space(ex.get("question", ""))
        passage_tokens = tokenize(context)[: config["max_context_tokens"]]
        evidence = select_evidence_sentence(question, context).text
        evidence_start = context.find(evidence) if evidence else -1
        evidence_end = evidence_start + len(evidence) if evidence_start >= 0 else -1
        start_flags, end_flags = token_feature_flags(
            passage_tokens,
            context,
            evidence_start if evidence_start >= 0 else None,
            evidence_end if evidence_end > evidence_start else None,
        )
        start_flags = start_flags[: config["max_context_tokens"]] + [0] * (config["max_context_tokens"] - len(start_flags))
        end_flags = end_flags[: config["max_context_tokens"]] + [0] * (config["max_context_tokens"] - len(end_flags))
        batch = {
            "passage_ids": torch.tensor([encode(passage_tokens, vocab, config["max_context_tokens"])], dtype=torch.long, device=device),
            "question_ids": torch.tensor([encode(tokenize(question), vocab, config["max_question_tokens"])], dtype=torch.long, device=device),
            "start_features": torch.tensor([start_flags[: config["max_context_tokens"]]], dtype=torch.long, device=device),
            "end_features": torch.tensor([end_flags[: config["max_context_tokens"]]], dtype=torch.long, device=device),
        }
        with torch.no_grad():
            output = model(**batch, max_answer_len=config["max_answer_tokens"])
        words = []
        for idx in output.logits[0].argmax(dim=-1).detach().cpu().tolist():
            word = inv_vocab.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in {"<pad>", "<bos>"}:
                words.append(word)
        rows.append({"id": ex.get("id"), "question": question, "reference": ex.get("answer", ""), "prediction": " ".join(words), "model": "S-NET"})
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
