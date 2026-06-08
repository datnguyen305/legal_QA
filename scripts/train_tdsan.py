#!/usr/bin/env python3
"""Train TD-SAN / DynSAN for extractive multi-passage Legal QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data_preprocessing.tdsan_preprocess import load_tdsan_records


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def toks(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_vocab(records: list[dict], min_freq: int = 1) -> dict[str, int]:
    from collections import Counter

    counter = Counter()
    for row in records:
        counter.update(toks(row["question"]))
        for passage in row["passages"]:
            counter.update(toks(passage))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default="models/tdsan")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-passages", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--max-passage-tokens", type=int, default=1536)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("TD-SAN training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.tdsan_model import TdsanForQuestionAnswering

    train_records = load_tdsan_records(
        args.train_data, args.context_dir, args.train_limit, args.max_passages, args.max_context_chars
    )
    dev_records = load_tdsan_records(
        args.dev_data, args.context_dir, args.dev_limit, args.max_passages, args.max_context_chars
    )
    if not train_records or not dev_records:
        raise SystemExit("No TD-SAN records were created. Check context files and answer spans.")
    vocab = build_vocab(train_records + dev_records)

    class TdsanDataset(Dataset):
        def __init__(self, records: list[dict]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            row = self.records[idx]
            question_tokens = toks(row["question"])
            passage_text = " ".join(row["passages"])
            passage_tokens = toks(passage_text)
            prefix = passage_text[: row["answer_start"]]
            answer = passage_text[row["answer_start"] : row["answer_end"]]
            start = min(len(toks(prefix)), args.max_passage_tokens - 1)
            end = min(start + max(1, len(toks(answer))) - 1, args.max_passage_tokens - 1)
            passage_ids = encode(passage_tokens, vocab, args.max_passage_tokens)
            rank_ids = []
            cursor = 0
            for rank, passage in enumerate(row["passages"]):
                count = len(toks(passage))
                rank_ids.extend([rank] * count)
                cursor += count
            rank_ids = (rank_ids[: args.max_passage_tokens] + [0] * args.max_passage_tokens)[: args.max_passage_tokens]
            return {
                "question_ids": torch.tensor(encode(question_tokens, vocab, args.max_question_tokens), dtype=torch.long),
                "passage_ids": torch.tensor(passage_ids, dtype=torch.long),
                "passage_rank_ids": torch.tensor(rank_ids, dtype=torch.long),
                "start_positions": torch.tensor(start, dtype=torch.long),
                "end_positions": torch.tensor(end, dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TdsanForQuestionAnswering(
        vocab_size=len(vocab),
        pad_token_id=vocab["<pad>"],
        hidden=args.hidden,
        heads=args.heads,
        top_k=args.top_k,
        max_position=max(args.max_question_tokens, args.max_passage_tokens) + 8,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(TdsanDataset(train_records), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TdsanDataset(dev_records), batch_size=args.batch_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_dev = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
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
                batch = {k: v.to(device) for k, v in batch.items()}
                dev_loss += float(model(**batch).loss.detach().cpu())
        dev_loss /= max(1, len(dev_loader))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            with (Path(args.output_dir) / "tdsan_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"vocab_size": len(vocab), "pad_token_id": vocab["<pad>"]}, f, indent=2)
            with (Path(args.output_dir) / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
