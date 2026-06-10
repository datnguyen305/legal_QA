#!/usr/bin/env python3
"""Train Simple Curriculum Pointer-Generator for long-context QA."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_preprocessing.cpg_preprocess import load_cpg_records
from data_preprocessing.qa_preprocess import tokenize


SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_vocab(records: list[dict], min_freq: int) -> dict[str, int]:
    counter = Counter()
    for row in records:
        counter.update(tokenize(row["question"]))
        counter.update(tokenize(row["context"]))
        counter.update(tokenize(row["answer"]))
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_len: int, add_bos_eos: bool = False) -> list[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    if add_bos_eos:
        ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]
    ids = ids[:max_len]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default="models/cpg")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--chunk-sizes", default="50,100,200,500")
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--easy-ratio", type=float, default=1.0)
    parser.add_argument("--easy-ratio-decay", type=float, default=0.25)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Curriculum pointer-generator training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.cpg_model import CurriculumPointerGenerator

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    train_records = load_cpg_records(
        args.train_data,
        args.context_dir,
        args.train_limit,
        chunk_sizes,
        args.max_context_tokens,
        args.easy_ratio,
        progress_label="initial train",
    )
    dev_records = load_cpg_records(
        args.dev_data,
        args.context_dir,
        args.dev_limit,
        chunk_sizes,
        args.max_context_tokens,
        0.5,
        progress_label="dev",
    )
    if not train_records or not dev_records:
        raise SystemExit("No curriculum pointer-generator records were created.")
    vocab = build_vocab(train_records + dev_records, args.min_freq)

    class CpgDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            return {
                "context_ids": torch.tensor(encode(tokenize(row["context"]), vocab, args.max_context_tokens), dtype=torch.long),
                "question_ids": torch.tensor(encode(tokenize(row["question"]), vocab, args.max_question_tokens), dtype=torch.long),
                "answer_ids": torch.tensor(
                    encode(tokenize(row["answer"]), vocab, args.max_answer_tokens, add_bos_eos=True),
                    dtype=torch.long,
                ),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CurriculumPointerGenerator(
        len(vocab),
        vocab["<pad>"],
        vocab["<unk>"],
        vocab["<bos>"],
        vocab["<eos>"],
        hidden=args.hidden,
        decoder_hidden=args.decoder_hidden,
        block_size=args.block_size,
    ).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.startswith("cuda")}
    dev_loader = DataLoader(CpgDataset(dev_records), batch_size=args.batch_size, **loader_kwargs)
    best_dev = None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        easy_ratio = max(0.0, args.easy_ratio - (epoch - 1) * args.easy_ratio_decay)
        train_records_epoch = load_cpg_records(
            args.train_data,
            args.context_dir,
            args.train_limit,
            chunk_sizes,
            args.max_context_tokens,
            easy_ratio,
            seed=23 + epoch,
            progress_label=f"train epoch {epoch}",
        )
        train_loader = DataLoader(CpgDataset(train_records_epoch), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(out.loss.detach().cpu())
            if step % 100 == 0:
                print(f"epoch={epoch} easy_ratio={easy_ratio:.2f} step={step}/{len(train_loader)} loss={total / step:.4f}")
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
            with (Path(args.output_dir) / "cpg_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"vocab_size": len(vocab)}, f, indent=2)
            with (Path(args.output_dir) / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
