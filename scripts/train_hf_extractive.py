#!/usr/bin/env python3
"""Fine-tune pretrained encoder models for extractive Legal QA."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.cpg_preprocess import progress_bar
from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import make_extractive_record
from evaluate_predictions import rouge_l


def extend_position_capacity(model, tokenizer, max_length: int) -> None:
    """Extend RoBERTa-style position embeddings/buffers for longer inputs."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return

    base = getattr(model, "roberta", None)
    embeddings = getattr(base, "embeddings", None)
    position_embeddings = getattr(embeddings, "position_embeddings", None)
    if embeddings is None or position_embeddings is None:
        return

    padding_idx = position_embeddings.padding_idx
    if padding_idx is None:
        padding_idx = tokenizer.pad_token_id or 0
    required_positions = max_length + padding_idx + 1
    if required_positions <= position_embeddings.num_embeddings:
        return

    old_weight = position_embeddings.weight.data
    new_embeddings = nn.Embedding(
        required_positions,
        position_embeddings.embedding_dim,
        padding_idx=position_embeddings.padding_idx,
    ).to(device=old_weight.device, dtype=old_weight.dtype)
    new_embeddings.weight.data[: old_weight.size(0)] = old_weight
    new_embeddings.weight.data[old_weight.size(0) :] = old_weight[-1].unsqueeze(0)
    embeddings.position_embeddings = new_embeddings
    embeddings.register_buffer("position_ids", torch.arange(required_positions, device=old_weight.device).expand((1, -1)), persistent=False)
    embeddings.register_buffer("token_type_ids", torch.zeros((1, required_positions), dtype=torch.long, device=old_weight.device), persistent=False)
    model.config.max_position_embeddings = required_positions
    print(f"Extended position embeddings to {required_positions} for max_length={max_length}", flush=True)


def find_subsequence_ids(tokens: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(tokens):
        return None
    for i in range(len(tokens) - len(needle) + 1):
        if tokens[i : i + len(needle)] == needle:
            return i
    return None


def pair_special_count(tokenizer, question_ids: list[int]) -> int:
    probe = tokenizer.build_inputs_with_special_tokens(question_ids, [tokenizer.unk_token_id or 0])
    return len(probe) - len(question_ids) - 1


def slow_train_feature(row: dict, tokenizer, max_length: int, doc_stride: int) -> dict | None:
    question_ids = tokenizer.encode(row["question"], add_special_tokens=False)[: min(128, max_length // 3)]
    context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
    answer_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
    answer_start = find_subsequence_ids(context_ids, answer_ids)
    if answer_start is None:
        return None
    answer_end = answer_start + len(answer_ids) - 1
    max_context_len = max(1, max_length - len(question_ids) - pair_special_count(tokenizer, question_ids))
    window_start = max(0, min(answer_start, answer_end - max_context_len + 1))
    window_end = min(len(context_ids), window_start + max_context_len)
    window_ids = context_ids[window_start:window_end]
    input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, window_ids)
    context_offset = find_subsequence_ids(input_ids, window_ids)
    if context_offset is None or answer_end >= window_end:
        return None
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(question_ids, window_ids)
    pad_len = max_length - len(input_ids)
    if pad_len < 0:
        return None
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
    token_type_ids = token_type_ids + [0] * pad_len if token_type_ids else None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "start_positions": context_offset + answer_start - window_start,
        "end_positions": context_offset + answer_end - window_start,
        "row_index": None,
    }


def slow_inference_features(row: dict, tokenizer, max_length: int, doc_stride: int) -> list[dict]:
    question_ids = tokenizer.encode(row["question"], add_special_tokens=False)[: min(128, max_length // 3)]
    context_ids = tokenizer.encode(row["context"], add_special_tokens=False)
    max_context_len = max(1, max_length - len(question_ids) - pair_special_count(tokenizer, question_ids))
    step = max(1, max_context_len - doc_stride)
    features = []
    for window_start in range(0, max(1, len(context_ids)), step):
        window_ids = context_ids[window_start : window_start + max_context_len]
        if not window_ids:
            break
        input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, window_ids)
        context_offset = find_subsequence_ids(input_ids, window_ids)
        if context_offset is None:
            continue
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(question_ids, window_ids)
        pad_len = max_length - len(input_ids)
        if pad_len < 0:
            continue
        features.append(
            {
                "input_ids": input_ids + [tokenizer.pad_token_id] * pad_len,
                "attention_mask": [1] * (max_length - pad_len) + [0] * pad_len,
                "token_type_ids": token_type_ids + [0] * pad_len if token_type_ids else None,
                "context_offset": context_offset,
                "window_ids": window_ids,
            }
        )
        if window_start + max_context_len >= len(context_ids):
            break
    return features


def build_records(path: str, context_dir: str, limit: int | None, max_context_chars: int, progress_label: str) -> list[dict]:
    examples = load_examples(path, limit)
    rows = []
    total = len(examples)
    print(f"Loading HF extractive records for {progress_label}: {total} examples", flush=True)
    for idx, ex in enumerate(examples, start=1):
        row = make_extractive_record(ex, context_dir, max_context_chars)
        if row is not None:
            rows.append(row)
        if idx == total or idx % 500 == 0:
            progress_bar(f"Preprocess HF extractive {progress_label}", idx, total, len(rows))
    return rows


def prepare_features(rows: list[dict], tokenizer, max_length: int, doc_stride: int, progress_label: str) -> list[dict]:
    features = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        if not tokenizer.is_fast:
            feature = slow_train_feature(row, tokenizer, max_length, doc_stride)
            if feature is not None:
                feature["row_index"] = idx - 1
                features.append(feature)
            if idx == total or idx % 500 == 0:
                progress_bar(f"Tokenize HF extractive {progress_label}", idx, total, len(features))
            continue
        tokenized = tokenizer(
            row["question"],
            row["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            offsets = tokenized["offset_mapping"][i]
            context_indexes = [j for j, sid in enumerate(sequence_ids) if sid == 1]
            if not context_indexes:
                continue
            ctx_start = context_indexes[0]
            ctx_end = context_indexes[-1]
            answer_start = row["answer_start"]
            answer_end = row["answer_end"]
            if offsets[ctx_start][0] > answer_start or offsets[ctx_end][1] < answer_end:
                continue
            start_position = ctx_start
            while start_position <= ctx_end and offsets[start_position][0] <= answer_start:
                start_position += 1
            start_position -= 1
            end_position = ctx_end
            while end_position >= ctx_start and offsets[end_position][1] >= answer_end:
                end_position -= 1
            end_position += 1
            features.append(
                {
                    "input_ids": tokenized["input_ids"][i],
                    "attention_mask": tokenized["attention_mask"][i],
                    "token_type_ids": tokenized.get("token_type_ids", [None] * len(tokenized["input_ids"]))[i],
                    "start_positions": start_position,
                    "end_positions": end_position,
                    "row_index": idx - 1,
                }
            )
            break
        if idx == total or idx % 500 == 0:
            progress_bar(f"Tokenize HF extractive {progress_label}", idx, total, len(features))
    return features


def decode_best_span(row: dict, tokenizer, model, device, max_length: int, doc_stride: int, max_answer_tokens: int) -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("HF extractive inference requires PyTorch.") from exc

    if not tokenizer.is_fast:
        features = slow_inference_features(row, tokenizer, max_length, doc_stride)
        if not features:
            return ""
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long, device=device),
        }
        if features[0]["token_type_ids"] is not None and getattr(model.config, "type_vocab_size", 0) > 1:
            batch["token_type_ids"] = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(**batch)
        best_score = None
        best_text = ""
        for feature_idx, feature in enumerate(features):
            start_scores = out.start_logits[feature_idx]
            end_scores = out.end_logits[feature_idx]
            context_offset = feature["context_offset"]
            window_ids = feature["window_ids"]
            top_starts = torch.topk(start_scores, k=min(20, start_scores.numel())).indices.tolist()
            top_ends = torch.topk(end_scores, k=min(20, end_scores.numel())).indices.tolist()
            for start in top_starts:
                for end in top_ends:
                    local_start = start - context_offset
                    local_end = end - context_offset
                    if local_start < 0 or local_end < local_start or local_end >= len(window_ids):
                        continue
                    if local_end - local_start + 1 > max_answer_tokens:
                        continue
                    score = float(start_scores[start] + end_scores[end])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_text = tokenizer.decode(window_ids[local_start : local_end + 1], skip_special_tokens=True)
        return best_text

    tokenized = tokenizer(
        row["question"],
        row["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offset_mapping = tokenized.pop("offset_mapping")
    if "overflow_to_sample_mapping" in tokenized:
        tokenized.pop("overflow_to_sample_mapping")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    if "token_type_ids" in tokenized and (tokenized["token_type_ids"] is None or getattr(model.config, "type_vocab_size", 0) <= 1):
        tokenized.pop("token_type_ids")
    with torch.no_grad():
        out = model(**tokenized)
    best_score = None
    best_text = ""
    for feature_idx in range(out.start_logits.size(0)):
        start_scores = out.start_logits[feature_idx]
        end_scores = out.end_logits[feature_idx]
        offsets = offset_mapping[feature_idx].tolist()
        top_starts = torch.topk(start_scores, k=min(20, start_scores.numel())).indices.tolist()
        top_ends = torch.topk(end_scores, k=min(20, end_scores.numel())).indices.tolist()
        for start in top_starts:
            for end in top_ends:
                if end < start or end - start + 1 > max_answer_tokens:
                    continue
                char_start, _ = offsets[start]
                _, char_end = offsets[end]
                if char_end <= char_start:
                    continue
                score = float(start_scores[start] + end_scores[end])
                if best_score is None or score > best_score:
                    best_score = score
                    best_text = row["context"][char_start:char_end]
    return best_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("HF extractive training requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        print(f"{args.model_name} tokenizer is slow; using token-span fallback without offset mappings.", flush=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    extend_position_capacity(model, tokenizer, args.max_length)
    train_rows = build_records(args.train_data, args.context_dir, args.train_limit, args.max_context_chars, "train")
    dev_rows = build_records(args.dev_data, args.context_dir, args.dev_limit, args.max_context_chars, "dev")
    train_features = prepare_features(train_rows, tokenizer, args.max_length, args.doc_stride, "train")
    dev_features = prepare_features(dev_rows, tokenizer, args.max_length, args.doc_stride, "dev")
    if getattr(model.config, "type_vocab_size", 0) <= 1:
        for feature in train_features + dev_features:
            feature["token_type_ids"] = None
    if not train_features or not dev_features:
        raise SystemExit("No tokenized HF extractive features were created.")

    class FeatureDataset(Dataset):
        def __init__(self, features: list[dict]) -> None:
            self.features = features

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, idx: int) -> dict:
            feature = self.features[idx]
            item = {
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
                "start_positions": torch.tensor(feature["start_positions"], dtype=torch.long),
                "end_positions": torch.tensor(feature["end_positions"], dtype=torch.long),
                "row_index": torch.tensor(feature["row_index"], dtype=torch.long),
            }
            if feature["token_type_ids"] is not None:
                item["token_type_ids"] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
            return item

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = args.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    train_loader = DataLoader(FeatureDataset(train_features), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(FeatureDataset(dev_features), batch_size=args.batch_size, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_rouge = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("row_index")
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_context():
                out = model(**batch)
                loss = out.loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += float(loss.detach().float().cpu())
            if step == len(train_loader) or step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total / step:.4f}", flush=True)
        model.eval()
        dev_loss = 0.0
        dev_predictions = []
        dev_refs = []
        with torch.no_grad():
            for step, batch in enumerate(dev_loader, start=1):
                idxs = batch.pop("row_index").tolist()
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_context():
                    out = model(**batch)
                dev_loss += float(out.loss.detach().float().cpu())
                if step == len(dev_loader) or step % 50 == 0:
                    progress_bar(f"Score HF extractive dev epoch {epoch}", step, len(dev_loader), step)
        for idx, row in enumerate(dev_rows, start=1):
            dev_predictions.append(decode_best_span(row, tokenizer, model, device, args.max_length, args.doc_stride, args.max_answer_tokens))
            dev_refs.append(row["answer"])
            if idx == len(dev_rows) or idx % 500 == 0:
                progress_bar(f"Generate HF extractive dev epoch {epoch}", idx, len(dev_rows), idx)
        dev_loss /= max(1, len(dev_loader))
        dev_rouge = sum(rouge_l(p, r) for p, r in zip(dev_predictions, dev_refs)) / max(1, len(dev_refs))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f} dev_rouge_l={dev_rouge:.4f}")
        if best_dev_rouge is None or dev_rouge > best_dev_rouge:
            best_dev_rouge = dev_rouge
            bad_epochs = 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            with (out_dir / "hf_extractive_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1
            print(f"dev ROUGE-L did not improve for {bad_epochs}/{args.patience} epochs")
            if bad_epochs >= args.patience:
                break


if __name__ == "__main__":
    main()
