#!/usr/bin/env python3
"""Train the Deep Cascade Model for multi-document Legal QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data_preprocessing.deep_cascade_preprocess import load_deep_cascade_records


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def toks(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_vocab(records: list[dict]) -> dict[str, int]:
    from collections import Counter

    counter = Counter()
    for row in records:
        counter.update(toks(row["question"]))
        counter.update(toks(row["context"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common():
        if token not in vocab:
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
    parser.add_argument("--output-dir", default="models/deep_cascade")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=4)
    parser.add_argument("--max-paragraphs", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ranker-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Deep Cascade training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.deep_cascade_model import DeepCascadeReader, FeatureRanker

    train_records = load_deep_cascade_records(
        args.train_data, args.context_dir, args.train_limit, args.max_docs, args.max_paragraphs, args.max_context_chars
    )
    dev_records = load_deep_cascade_records(
        args.dev_data, args.context_dir, args.dev_limit, args.max_docs, args.max_paragraphs, args.max_context_chars
    )
    if not train_records or not dev_records:
        raise SystemExit("No Deep Cascade records were created. Check contexts and answer spans.")
    vocab = build_vocab(train_records + dev_records)

    class RankDataset(Dataset):
        def __init__(self, records: list[dict], level: str) -> None:
            self.rows = []
            for row in records:
                if level == "doc":
                    for feat, label in zip(row["doc_features"], row["doc_labels"]):
                        self.rows.append((feat, label))
                else:
                    for feats, labels in zip(row["para_features"], row["para_labels"]):
                        for feat, label in zip(feats, labels):
                            self.rows.append((feat, label))

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            feat, label = self.rows[idx]
            return {"features": torch.tensor(feat, dtype=torch.float), "label": torch.tensor(label, dtype=torch.float)}

    class ReaderDataset(Dataset):
        def __init__(self, records: list[dict]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            row = self.records[idx]
            context_tokens = toks(row["context"])
            question_tokens = toks(row["question"])
            prefix = row["context"][: row["answer_start"]]
            answer = row["context"][row["answer_start"] : row["answer_end"]]
            start = min(len(toks(prefix)), args.max_context_tokens - 1)
            end = min(start + max(1, len(toks(answer))) - 1, args.max_context_tokens - 1)
            token_features = []
            qset = set(question_tokens)
            for token in context_tokens[: args.max_context_tokens]:
                token_features.append([1.0 if token in qset else 0.0, 1.0 if token in {".", "!", "?", ";"} else 0.0, 0.0, 0.0, 0.0])
            token_features += [[0.0] * 5 for _ in range(args.max_context_tokens - len(token_features))]

            doc_spans = []
            para_spans = []
            cursor_tokens = 0
            for doc in row["docs"][: args.max_docs]:
                doc_start = cursor_tokens
                doc_para_spans = []
                for para in doc["paragraphs"][: args.max_paragraphs]:
                    p_len = len(toks(para["text"]))
                    p_start = min(cursor_tokens, args.max_context_tokens - 1)
                    p_end = min(cursor_tokens + p_len - 1, args.max_context_tokens - 1)
                    doc_para_spans.append([p_start, p_end] if p_len > 0 and p_start <= p_end else [-1, -1])
                    cursor_tokens += p_len
                doc_end = min(max(doc_start, cursor_tokens - 1), args.max_context_tokens - 1)
                doc_spans.append([min(doc_start, args.max_context_tokens - 1), doc_end])
                doc_para_spans += [[-1, -1] for _ in range(args.max_paragraphs - len(doc_para_spans))]
                para_spans.append(doc_para_spans)
            doc_spans += [[-1, -1] for _ in range(args.max_docs - len(doc_spans))]
            para_spans += [[[-1, -1] for _ in range(args.max_paragraphs)] for _ in range(args.max_docs - len(para_spans))]
            return {
                "question_ids": torch.tensor(encode(question_tokens, vocab, args.max_question_tokens), dtype=torch.long),
                "context_ids": torch.tensor(encode(context_tokens, vocab, args.max_context_tokens), dtype=torch.long),
                "token_features": torch.tensor(token_features, dtype=torch.float),
                "doc_spans": torch.tensor(doc_spans, dtype=torch.long),
                "para_spans": torch.tensor(para_spans, dtype=torch.long),
                "doc_features": torch.tensor(row["doc_features"], dtype=torch.float),
                "para_features": torch.tensor(row["para_features"], dtype=torch.float),
                "doc_labels": torch.tensor(row["doc_labels"], dtype=torch.float),
                "para_labels": torch.tensor(row["para_labels"], dtype=torch.float),
                "start_positions": torch.tensor(start, dtype=torch.long),
                "end_positions": torch.tensor(end, dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.startswith("cuda")}
    doc_ranker = FeatureRanker().to(device)
    para_ranker = FeatureRanker().to(device)

    def train_ranker(model: FeatureRanker, dataset: Dataset, name: str) -> None:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.ranker_epochs):
            total = 0.0
            for batch in loader:
                feat = batch["features"].to(device)
                label = batch["label"].to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(feat), label)
                loss.backward()
                opt.step()
                opt.zero_grad()
                total += float(loss.detach().cpu())
            print(f"{name}_ranker epoch={epoch + 1} loss={total / max(1, len(loader)):.4f}")

    train_ranker(doc_ranker, RankDataset(train_records, "doc"), "doc")
    train_ranker(para_ranker, RankDataset(train_records, "para"), "para")

    reader = DeepCascadeReader(len(vocab), vocab["<pad>"], hidden=args.hidden).to(device)
    train_loader = DataLoader(ReaderDataset(train_records), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(ReaderDataset(dev_records), batch_size=args.batch_size, **loader_kwargs)
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_dev = None
    for epoch in range(1, args.epochs + 1):
        reader.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = reader(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(out.loss.detach().cpu())
            if step % 100 == 0:
                print(f"reader epoch={epoch} step={step}/{len(train_loader)} loss={total / step:.4f}")
        reader.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                dev_loss += float(reader(**batch).loss.detach().cpu())
        dev_loss /= max(1, len(dev_loader))
        print(f"reader epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(doc_ranker.state_dict(), Path(args.output_dir) / "doc_ranker.bin")
            torch.save(para_ranker.state_dict(), Path(args.output_dir) / "para_ranker.bin")
            torch.save(reader.state_dict(), Path(args.output_dir) / "reader.bin")
            with (Path(args.output_dir) / "deep_cascade_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"vocab_size": len(vocab), "pad_token_id": vocab["<pad>"]}, f, indent=2)
            with (Path(args.output_dir) / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
