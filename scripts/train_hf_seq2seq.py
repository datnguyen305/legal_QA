#!/usr/bin/env python3
"""Fine-tune pretrained seq2seq models on Legal QA.

Works with ViT5, BARTpho, mT5, mBART, and other Hugging Face
AutoModelForSeq2SeqLM checkpoints.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.cpg_preprocess import progress_bar, sample_gold_context
from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import normalize_space
from evaluate_predictions import rouge_l


def disable_apex_import() -> None:
    """Force Transformers to avoid an installed but ABI-broken Apex package.

    Some notebook images ship an ``apex`` wheel whose fused CUDA extensions do
    not match the active PyTorch build. T5/mT5 can then crash while constructing
    layer norm. Raising ImportError for apex lets Transformers use the standard
    PyTorch implementation instead.
    """
    sys.modules["apex"] = None
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "apex" or name.startswith("apex."):
            raise ImportError("Apex import disabled by DISABLE_APEX=1")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def make_input(question: str, context: str) -> str:
    return f"Câu hỏi: {question}\nVăn bản pháp luật liên quan:\n{context}\nTrả lời:"


def build_rows(path: str, context_dir: str, limit: int | None, max_context_chars: int, progress_label: str | None = None) -> list[dict]:
    rows = []
    examples = load_examples(path, limit)
    total = len(examples)
    if progress_label:
        print(f"Loading seq2seq rows for {progress_label}: {total} examples", flush=True)
    for idx, ex in enumerate(examples, start=1):
        context = sample_gold_context(ex, context_dir)[:max_context_chars]
        question = normalize_space(ex.get("question", ""))
        answer = normalize_space(ex.get("answer", ""))
        if context and question and answer:
            rows.append({"id": ex.get("id"), "question": question, "context": context, "answer": answer})
        if progress_label and (idx == total or idx % 500 == 0):
            progress_bar(f"Preprocess seq2seq {progress_label}", idx, total, len(rows))
    return rows


def configure_tokenizer(tokenizer, model_name: str, src_lang: str | None, tgt_lang: str | None) -> int | None:
    lower = model_name.lower()
    forced_bos_token_id = None
    if src_lang and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if tgt_lang and hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang
    if tgt_lang and "mbart" in lower and hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return forced_bos_token_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Hugging Face checkpoint, e.g. VietAI/vit5-base")
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-pretokenize", action="store_true", help="Tokenize lazily in __getitem__ instead of caching tensors before training.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dev-generate-batch-size", type=int, default=None)
    parser.add_argument("--dev-eval-limit", type=int, default=None, help="Limit dev examples used for ROUGE-L early stopping.")
    parser.add_argument("--dev-num-beams", type=int, default=1, help="Beams for dev ROUGE-L early stopping generation.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--src-lang", default=None, help="For mBART, e.g. vi_VN")
    parser.add_argument("--tgt-lang", default=None, help="For mBART, e.g. vi_VN")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        if os.environ.get("DISABLE_APEX", "1") == "1":
            disable_apex_import()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Pretrained seq2seq training requires torch and transformers.") from exc

    train_rows = build_rows(args.train_data, args.context_dir, args.train_limit, args.max_context_chars, progress_label="train")
    dev_rows = build_rows(args.dev_data, args.context_dir, args.dev_limit, args.max_context_chars, progress_label="dev")
    if not train_rows or not dev_rows:
        raise SystemExit("No seq2seq rows were created.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    forced_bos_token_id = configure_tokenizer(tokenizer, args.model_name, args.src_lang, args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    class Seq2SeqDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows
            self.items = None
            if not args.no_pretokenize:
                self.items = []
                total = len(rows)
                for idx, row in enumerate(rows, start=1):
                    self.items.append(self.encode_row(row))
                    if idx == total or idx % 500 == 0:
                        progress_bar("Tokenize seq2seq dataset", idx, total, idx)

        def __len__(self) -> int:
            return len(self.rows)

        def encode_row(self, row: dict) -> dict:
            source = make_input(row["question"], row["context"])
            encoded = tokenizer(
                source,
                max_length=args.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = tokenizer(
                text_target=row["answer"],
                max_length=args.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"][0]
            labels[labels == tokenizer.pad_token_id] = -100
            item = {key: value[0] for key, value in encoded.items()}
            item["labels"] = labels
            return item

        def __getitem__(self, idx: int) -> dict:
            if self.items is not None:
                return self.items[idx]
            return self.encode_row(self.rows[idx])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    model.to(device)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    print("Building tokenized train dataset", flush=True)
    train_dataset = Seq2SeqDataset(train_rows)
    print("Building tokenized dev dataset", flush=True)
    dev_dataset = Seq2SeqDataset(dev_rows)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, **loader_kwargs)
    dev_gen_batch_size = args.dev_generate_batch_size or args.batch_size
    dev_gen_rows = dev_rows[: args.dev_eval_limit] if args.dev_eval_limit is not None else dev_rows
    print(
        f"Dev ROUGE-L early stopping uses {len(dev_gen_rows)}/{len(dev_rows)} dev examples "
        f"with num_beams={args.dev_num_beams}",
        flush=True,
    )
    dev_gen_loader = DataLoader(dev_gen_rows, batch_size=dev_gen_batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps)
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_dev_rouge = None
    bad_epochs = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context():
                output = model(**batch)
                loss = output.loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total += float(loss.detach().float().cpu()) * args.grad_accum_steps
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total / step:.4f}")

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast_context():
                    dev_loss += float(model(**batch).loss.detach().float().cpu())
        dev_loss /= max(1, len(dev_loader))

        predictions = []
        references = []
        with torch.no_grad():
            for batch_idx, rows in enumerate(dev_gen_loader, start=1):
                sources = [make_input(q, c) for q, c in zip(rows["question"], rows["context"])]
                encoded = tokenizer(
                    sources,
                    max_length=args.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                generate_kwargs = {
                    "max_new_tokens": args.max_target_length,
                    "num_beams": args.dev_num_beams,
                }
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
                ids = model.generate(**encoded, **generate_kwargs)
                predictions.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
                references.extend(rows["answer"])
                if batch_idx == len(dev_gen_loader) or batch_idx % 20 == 0:
                    progress_bar("Generate dev seq2seq", batch_idx, len(dev_gen_loader), len(predictions))
        dev_rouge_l = sum(rouge_l(pred, ref) for pred, ref in zip(predictions, references)) / max(1, len(predictions))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f} dev_rouge_l={dev_rouge_l:.4f}")

        if best_dev_rouge is None or dev_rouge_l > best_dev_rouge:
            best_dev_rouge = dev_rouge_l
            bad_epochs = 0
            model.save_pretrained(args.output_dir, safe_serialization=False)
            tokenizer.save_pretrained(args.output_dir)
            with (Path(args.output_dir) / "seq2seq_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"forced_bos_token_id": forced_bos_token_id}, f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1
            print(f"dev ROUGE-L did not improve for {bad_epochs}/{args.patience} epochs")
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_dev_rouge_l={best_dev_rouge:.4f}")
                break


if __name__ == "__main__":
    main()
