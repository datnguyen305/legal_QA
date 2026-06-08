#!/usr/bin/env python3
"""Fine-tune the EQUALS BERT-style MRC component."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import make_extractive_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--base-model", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/equals_mrc")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training EQUALS MRC requires: python3 -m pip install datasets -r requirements-models.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def load_records(path: str, limit: int | None) -> list[dict]:
        records = []
        for example in load_examples(path, limit):
            record = make_extractive_record(
                example,
                context_dir=args.context_dir,
                max_context_chars=args.max_context_chars,
                prefer_article=True,
            )
            if record is not None:
                records.append(record)
        return records

    train_records = load_records(args.train_data, args.train_limit)
    dev_records = load_records(args.dev_data, args.dev_limit)
    if not train_records or not dev_records:
        raise SystemExit(
            "No extractive QA records were created. Check context files and answer spans, "
            "or increase --max-context-chars."
        )

    def preprocess(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offsets = tokenized.pop("offset_mapping")
        starts = []
        ends = []
        for i, offset in enumerate(offsets):
            sample_idx = sample_map[i]
            answer_start = batch["answer_start"][sample_idx]
            answer_end = batch["answer_end"][sample_idx]
            sequence_ids = tokenized.sequence_ids(i)
            context_start = next(j for j, sid in enumerate(sequence_ids) if sid == 1)
            context_end = len(sequence_ids) - 1 - next(
                j for j, sid in enumerate(reversed(sequence_ids)) if sid == 1
            )
            if offset[context_start][0] > answer_start or offset[context_end][1] < answer_end:
                starts.append(0)
                ends.append(0)
                continue
            start_pos = context_start
            while start_pos <= context_end and offset[start_pos][0] <= answer_start:
                start_pos += 1
            end_pos = context_end
            while end_pos >= context_start and offset[end_pos][1] >= answer_end:
                end_pos -= 1
            starts.append(start_pos - 1)
            ends.append(end_pos + 1)
        tokenized["start_positions"] = starts
        tokenized["end_positions"] = ends
        return tokenized

    train_dataset = Dataset.from_list(train_records).map(preprocess, batched=True, remove_columns=list(train_records[0]))
    dev_dataset = Dataset.from_list(dev_records).map(preprocess, batched=True, remove_columns=list(dev_records[0]))
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
