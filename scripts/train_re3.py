#!/usr/bin/env python3
"""Train RE3QA: retrieve, read, rerank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.re3_preprocess import load_re3_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--base-model", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/re3")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--window-tokens", type=int, default=320)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--early-layer", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("RE3QA training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.re3_model import Re3QA

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_rows = load_re3_segments(args.train_data, args.context_dir, args.train_limit, args.window_tokens, args.stride, args.max_context_chars)
    dev_rows = load_re3_segments(args.dev_data, args.context_dir, args.dev_limit, args.window_tokens, args.stride, args.max_context_chars)
    if not train_rows or not dev_rows:
        raise SystemExit("No RE3QA segments were created.")

    class Re3Dataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            enc = tokenizer(
                row["question"],
                row["segment"],
                truncation="only_second",
                max_length=args.max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            sequence_ids = enc.sequence_ids(0)
            start_pos = end_pos = 0
            if row["answer_start"] is not None:
                for i, (start, end) in enumerate(offsets):
                    if sequence_ids[i] == 1 and start <= row["answer_start"] < end:
                        start_pos = i
                    if sequence_ids[i] == 1 and start < row["answer_end"] <= end:
                        end_pos = i
                        break
            item = {key: val[0] for key, val in enc.items()}
            item["retrieve_labels"] = torch.tensor(row["has_answer"], dtype=torch.long)
            item["start_positions"] = torch.tensor(start_pos, dtype=torch.long)
            item["end_positions"] = torch.tensor(end_pos, dtype=torch.long)
            item["rerank_labels"] = torch.tensor(0, dtype=torch.long)
            return item

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Re3QA(args.base_model, early_layer=args.early_layer, max_candidates=args.max_candidates).to(device)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.startswith("cuda")}
    train_loader = DataLoader(Re3Dataset(train_rows), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(Re3Dataset(dev_rows), batch_size=args.batch_size, **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_dev = None
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
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            tokenizer.save_pretrained(args.output_dir)
            with (Path(args.output_dir) / "re3_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
