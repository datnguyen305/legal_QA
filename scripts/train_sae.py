#!/usr/bin/env python3
"""Train Select, Answer and Explain on the local Legal QA dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.sae_preprocess import build_context_pool, load_passage, make_sae_answer_record, sample_candidate_refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--base-model", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/sae")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-sentences", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--selector-epochs", type=int, default=1)
    parser.add_argument("--answer-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("SAE training requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.sae_model import SaeAnswerExplain, SaeDocumentSelector

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_examples = load_examples(args.train_data, args.train_limit)
    dev_examples = load_examples(args.dev_data, args.dev_limit)
    pool = build_context_pool(train_examples + dev_examples)

    class SelectorDataset(Dataset):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict:
            ex = self.examples[idx]
            refs, labels, scores = sample_candidate_refs(ex, pool, args.max_docs, rng)
            docs = [load_passage(args.context_dir, ref, args.max_context_chars) for ref in refs]
            encoded = tokenizer(
                [ex.get("question", "")] * len(docs),
                docs,
                truncation="only_second",
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value for key, value in encoded.items()}
            item["doc_labels"] = torch.tensor(labels, dtype=torch.float)
            item["doc_scores"] = torch.tensor(scores, dtype=torch.float)
            return item

    def answer_records(examples: list[dict]) -> list[dict]:
        records = []
        for ex in examples:
            record = make_sae_answer_record(
                ex,
                args.context_dir,
                args.max_context_chars,
                args.max_sentences,
            )
            if record is not None:
                records.append(record)
        return records

    class AnswerDataset(Dataset):
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
            support_labels = torch.zeros(args.max_sentences, dtype=torch.float)
            cursor = 0
            for j, sentence in enumerate(row["sentences"][: args.max_sentences]):
                char_start = row["context"].find(sentence, cursor)
                if char_start < 0:
                    char_start = cursor
                char_end = char_start + len(sentence)
                token_indices = [
                    tok_i
                    for tok_i, (start, end) in enumerate(offsets)
                    if sequence_ids[tok_i] == 1 and end > char_start and start < char_end
                ]
                if token_indices:
                    sentence_spans[j, 0] = token_indices[0]
                    sentence_spans[j, 1] = token_indices[-1]
                support_labels[j] = row["support"][j]
                cursor = char_end

            adjacency = torch.zeros(3, args.max_sentences, args.max_sentences)
            raw_adj = row["adjacency"]
            for rel in range(min(3, len(raw_adj))):
                for i in range(min(args.max_sentences, len(raw_adj[rel]))):
                    for j in range(min(args.max_sentences, len(raw_adj[rel][i]))):
                        adjacency[rel, i, j] = raw_adj[rel][i][j]

            item["sentence_spans"] = sentence_spans
            item["adjacency"] = adjacency
            item["support_labels"] = support_labels
            item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
            item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
            item["answer_type"] = torch.tensor(row["answer_type"], dtype=torch.long)
            return item

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    selector = SaeDocumentSelector(args.base_model, max_docs=args.max_docs).to(device)
    selector_loader = DataLoader(SelectorDataset(train_examples), batch_size=args.batch_size, shuffle=True)
    selector_opt = torch.optim.AdamW(selector.parameters(), lr=args.lr)
    selector_steps = max(1, len(selector_loader) * args.selector_epochs)
    selector_sched = get_linear_schedule_with_warmup(selector_opt, int(selector_steps * 0.1), selector_steps)
    for epoch in range(1, args.selector_epochs + 1):
        selector.train()
        total = 0.0
        for step, batch in enumerate(selector_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = selector(**batch)
            output.loss.backward()
            selector_opt.step()
            selector_sched.step()
            selector_opt.zero_grad()
            total += float(output.loss.detach().cpu())
            if step % 100 == 0:
                print(f"selector epoch={epoch} step={step}/{len(selector_loader)} loss={total / step:.4f}")

    train_answer = answer_records(train_examples)
    dev_answer = answer_records(dev_examples)
    if not train_answer or not dev_answer:
        raise SystemExit("No SAE answer/explain records were created; check contexts and answer spans.")
    answer_model = SaeAnswerExplain(args.base_model, max_sentences=args.max_sentences).to(device)
    answer_loader = DataLoader(AnswerDataset(train_answer), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(AnswerDataset(dev_answer), batch_size=args.batch_size)
    answer_opt = torch.optim.AdamW(answer_model.parameters(), lr=args.lr)
    answer_steps = max(1, len(answer_loader) * args.answer_epochs)
    answer_sched = get_linear_schedule_with_warmup(answer_opt, int(answer_steps * 0.1), answer_steps)
    best_dev = None
    for epoch in range(1, args.answer_epochs + 1):
        answer_model.train()
        total = 0.0
        for step, batch in enumerate(answer_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = answer_model(**batch)
            output.loss.backward()
            answer_opt.step()
            answer_sched.step()
            answer_opt.zero_grad()
            total += float(output.loss.detach().cpu())
            if step % 100 == 0:
                print(f"answer epoch={epoch} step={step}/{len(answer_loader)} loss={total / step:.4f}")
        answer_model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                dev_loss += float(answer_model(**batch).loss.detach().cpu())
        dev_loss /= max(1, len(dev_loader))
        print(f"answer epoch={epoch} train_loss={total / max(1, len(answer_loader)):.4f} dev_loss={dev_loss:.4f}")
        if best_dev is None or dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(selector.state_dict(), Path(args.output_dir) / "selector.bin")
            torch.save(answer_model.state_dict(), Path(args.output_dir) / "answer_explain.bin")
            tokenizer.save_pretrained(args.output_dir)
            with (Path(args.output_dir) / "sae_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": args.base_model,
                        "max_docs": args.max_docs,
                        "top_k": args.top_k,
                        "max_length": args.max_length,
                        "max_sentences": args.max_sentences,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
