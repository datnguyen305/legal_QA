#!/usr/bin/env python3
"""Train S-NET extraction-then-synthesis model.

The extraction stage supplies evidence text. During training, gold
``answer_start``/``answer_end`` labels are used when available; otherwise a
lexical evidence extractor is used. The synthesis stage fine-tunes a Hugging
Face seq2seq model on question + evidence + passage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import normalize_space
from model_architectures.snet_model import select_evidence_sentence, snet_input


def evidence_for_example(example: dict) -> str:
    context = normalize_space(example.get("context", ""))
    start = example.get("answer_start")
    end = example.get("answer_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
        return context[start:end]
    return select_evidence_sentence(example.get("question", ""), context).text


def build_rows(path: str, limit: int | None, max_context_chars: int) -> list[dict]:
    rows = []
    for ex in load_examples(path, limit):
        question = normalize_space(ex.get("question", ""))
        context = normalize_space(ex.get("context", ""))[:max_context_chars]
        answer = normalize_space(ex.get("answer", ""))
        evidence = evidence_for_example(ex)
        if question and context and answer:
            rows.append({"input": snet_input(question, context, evidence), "answer": answer})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--base-model", default="google/mt5-small")
    parser.add_argument("--output-dir", default="models/snet")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("S-NET training requires torch and transformers.") from exc

    train_rows = build_rows(args.train_data, args.train_limit, args.max_context_chars)
    dev_rows = build_rows(args.dev_data, args.dev_limit, args.max_context_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    class SNetDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            inputs = tokenizer(row["input"], max_length=args.max_input_length, truncation=True, padding="max_length", return_tensors="pt")
            labels = tokenizer(text_target=row["answer"], max_length=args.max_target_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"][0]
            labels[labels == tokenizer.pad_token_id] = -100
            item = {k: v[0] for k, v in inputs.items()}
            item["labels"] = labels
            return item

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(SNetDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(SNetDataset(dev_rows), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    best_dev = None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += float(out.loss.detach().cpu())
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
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
