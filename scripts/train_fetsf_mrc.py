#!/usr/bin/env python3
"""Train the proposed FETSF-MRC model on the local Legal QA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import make_extractive_record, sentence_evidence_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--base-model", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/fetsf_mrc")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-sentences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("FETSF-MRC training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.fetsf_model import FetsfMRC

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def records(path: str, limit: int | None) -> list[dict]:
        out = []
        for ex in load_examples(path, limit):
            record = make_extractive_record(ex, args.context_dir, args.max_context_chars, prefer_article=True)
            if record is None:
                continue
            sentences, evidence = sentence_evidence_labels(
                record["context"],
                record["answer_start"],
                record["answer_end"],
            )
            record["sentences"] = sentences[: args.max_sentences]
            record["evidence"] = evidence[: args.max_sentences]
            out.append(record)
        return out

    class FetsfDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            encoded = tokenizer(
                row["question"],
                row["context"],
                truncation="only_second",
                max_length=args.max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            sequence_ids = encoded.sequence_ids(0)
            item = {key: value[0] for key, value in encoded.items()}

            start_token = end_token = 0
            for i, (start, end) in enumerate(offsets):
                if sequence_ids[i] == 1 and start <= row["answer_start"] < end:
                    start_token = i
                if sequence_ids[i] == 1 and start < row["answer_end"] <= end:
                    end_token = i
                    break

            sentence_spans = torch.full((args.max_sentences, 2), -1, dtype=torch.long)
            evidence_labels = torch.zeros(args.max_sentences, dtype=torch.float)
            cursor = 0
            for j, sentence in enumerate(row["sentences"][: args.max_sentences]):
                char_start = row["context"].find(sentence, cursor)
                if char_start < 0:
                    char_start = cursor
                char_end = char_start + len(sentence)
                token_indices = [
                    i
                    for i, (start, end) in enumerate(offsets)
                    if sequence_ids[i] == 1 and end > char_start and start < char_end
                ]
                if token_indices:
                    sentence_spans[j, 0] = token_indices[0]
                    sentence_spans[j, 1] = token_indices[-1]
                evidence_labels[j] = row["evidence"][j]
                cursor = char_end

            item["sentence_spans"] = sentence_spans
            item["evidence_labels"] = evidence_labels
            item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
            item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
            item["answer_type"] = torch.tensor(0, dtype=torch.long)
            return item

    train_rows = records(args.train_data, args.train_limit)
    dev_rows = records(args.dev_data, args.dev_limit)
    train_loader = DataLoader(FetsfDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(FetsfDataset(dev_rows), batch_size=args.batch_size)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FetsfMRC(args.base_model, max_sentences=args.max_sentences).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_dev = None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += float(output.loss.detach().cpu())
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={train_loss / step:.4f}")

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                dev_loss += float(output.loss.detach().cpu())
        dev_loss = dev_loss / max(1, len(dev_loader))
        print(f"epoch={epoch} train_loss={train_loss / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            tokenizer.save_pretrained(args.output_dir)
            with (Path(args.output_dir) / "fetsf_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": args.base_model,
                        "max_sentences": args.max_sentences,
                        "max_length": args.max_length,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
