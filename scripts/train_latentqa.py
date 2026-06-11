#!/usr/bin/env python3
"""Train LatentQA stochastic selector network."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.cpg_preprocess import sample_gold_context
from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from train_cpg import build_vocab, encode


def build_records(path: str, context_dir: str, limit: int | None, max_context_chars: int) -> list[dict]:
    rows = []
    for ex in load_examples(path, limit):
        context = sample_gold_context(ex, context_dir)[:max_context_chars]
        answer = normalize_space(ex.get("answer", ""))
        question = normalize_space(ex.get("question", ""))
        if context and answer and question:
            rows.append({"id": ex.get("id"), "question": question, "context": context, "answer": answer})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default="models/latentqa")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("LatentQA training requires PyTorch.") from exc

    from model_architectures.latentqa_model import LatentQA

    train_rows = build_records(args.train_data, args.context_dir, args.train_limit, args.max_context_chars)
    dev_rows = build_records(args.dev_data, args.context_dir, args.dev_limit, args.max_context_chars)
    if not train_rows or not dev_rows:
        train_probe = load_examples(args.train_data, 1)
        dev_probe = load_examples(args.dev_data, 1)
        train_keys = sorted(train_probe[0].keys()) if train_probe else []
        dev_keys = sorted(dev_probe[0].keys()) if dev_probe else []
        raise SystemExit(
            "No LatentQA records were created. LatentQA uses only the current sample's gold context: "
            "embedded sample text or the contexts[*].content file referenced by that sample. "
            f"First train keys={train_keys}, gold_context_found={bool(train_probe and sample_gold_context(train_probe[0], args.context_dir))}; "
            f"first dev keys={dev_keys}, gold_context_found={bool(dev_probe and sample_gold_context(dev_probe[0], args.context_dir))}."
        )
    vocab = build_vocab(train_rows + dev_rows, args.min_freq)

    class QADataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            return {
                "context_ids": torch.tensor(encode(tokenize(row["context"]), vocab, args.max_context_tokens), dtype=torch.long),
                "question_ids": torch.tensor(encode(tokenize(row["question"]), vocab, args.max_question_tokens), dtype=torch.long),
                "answer_ids": torch.tensor(encode(tokenize(row["answer"]), vocab, args.max_answer_tokens, True), dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    model = LatentQA(len(vocab), vocab["<pad>"], vocab["<bos>"], vocab["<eos>"], args.hidden, args.decoder_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(QADataset(train_rows), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(QADataset(dev_rows), batch_size=args.batch_size, **loader_kwargs)
    best_dev = None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_context():
                out = model(**batch)
                loss = out.loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            total += float(loss.detach().float().cpu())
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total / step:.4f}")
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_context():
                    loss = model(**batch).loss
                dev_loss += float(loss.detach().float().cpu())
        dev_loss /= max(1, len(dev_loader))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            json.dump(vars(args) | {"vocab_size": len(vocab)}, open(Path(args.output_dir) / "latentqa_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            json.dump(vocab, open(Path(args.output_dir) / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
