#!/usr/bin/env python3
"""Train S-NET GRU extraction-then-synthesis model from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from model_architectures.snet_model import SNetSynthesis, token_feature_flags
from train_cpg import build_vocab, encode


def build_rows(path: str, limit: int | None, max_context_chars: int) -> list[dict]:
    rows = []
    for ex in load_examples(path, limit):
        context = normalize_space(ex.get("context", ""))[:max_context_chars]
        question = normalize_space(ex.get("question", ""))
        answer = normalize_space(ex.get("answer", ""))
        if not context or not question or not answer:
            continue
        rows.append(
            {
                "id": ex.get("id"),
                "question": question,
                "context": context,
                "answer": answer,
                "answer_start": ex.get("answer_start"),
                "answer_end": ex.get("answer_end"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--output-dir", default="models/snet")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embed-size", type=int, default=300)
    parser.add_argument("--feature-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("S-NET training requires PyTorch.") from exc

    train_rows = build_rows(args.train_data, args.train_limit, args.max_context_chars)
    dev_rows = build_rows(args.dev_data, args.dev_limit, args.max_context_chars)
    if not train_rows or not dev_rows:
        raise SystemExit("No S-NET rows were created.")
    vocab = build_vocab(train_rows + dev_rows, args.min_freq)
    if len(vocab) > args.vocab_size:
        keep = {tok: idx for tok, idx in vocab.items() if idx < 4}
        for tok, _idx in sorted(vocab.items(), key=lambda item: item[1]):
            if tok not in keep:
                keep[tok] = len(keep)
            if len(keep) >= args.vocab_size:
                break
        vocab = keep

    class SNetDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            passage_tokens = tokenize(row["context"])[: args.max_context_tokens]
            start_flags, end_flags = token_feature_flags(
                passage_tokens,
                row["context"],
                row.get("answer_start") if isinstance(row.get("answer_start"), int) else None,
                row.get("answer_end") if isinstance(row.get("answer_end"), int) else None,
            )
            start_flags = start_flags[: args.max_context_tokens] + [0] * (args.max_context_tokens - len(start_flags))
            end_flags = end_flags[: args.max_context_tokens] + [0] * (args.max_context_tokens - len(end_flags))
            return {
                "passage_ids": torch.tensor(encode(passage_tokens, vocab, args.max_context_tokens), dtype=torch.long),
                "question_ids": torch.tensor(encode(tokenize(row["question"]), vocab, args.max_question_tokens), dtype=torch.long),
                "start_features": torch.tensor(start_flags[: args.max_context_tokens], dtype=torch.long),
                "end_features": torch.tensor(end_flags[: args.max_context_tokens], dtype=torch.long),
                "answer_ids": torch.tensor(encode(tokenize(row["answer"]), vocab, args.max_answer_tokens, True), dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SNetSynthesis(
        len(vocab),
        vocab["<pad>"],
        vocab["<bos>"],
        vocab["<eos>"],
        embed_size=args.embed_size,
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(SNetDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(SNetDataset(dev_rows), batch_size=args.batch_size)
    best_dev = None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(output.loss.detach().cpu())
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total / step:.4f}")
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                dev_loss += float(model(**batch).loss.detach().cpu())
        dev_loss /= max(1, len(dev_loader))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            with (Path(args.output_dir) / "snet_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"vocab_size": len(vocab)}, f, ensure_ascii=False, indent=2)
            with (Path(args.output_dir) / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
