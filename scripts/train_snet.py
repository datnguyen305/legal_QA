#!/usr/bin/env python3
"""Train S-NET GRU extraction-then-synthesis model from scratch."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.cpg_preprocess import progress_bar, sample_gold_context
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from evaluate_predictions import rouge_l
from model_architectures.snet_model import SNetSynthesis, token_feature_flags
from train_cpg import build_vocab, encode


def build_rows(path: str, context_dir: str, limit: int | None, max_context_chars: int, progress_label: str | None = None) -> list[dict]:
    rows = []
    examples = load_examples(path, limit)
    total = len(examples)
    if progress_label:
        print(f"Loading S-NET rows for {progress_label}: {total} examples", flush=True)
    for idx, ex in enumerate(examples, start=1):
        context = sample_gold_context(ex, context_dir)[:max_context_chars]
        question = normalize_space(ex.get("question", ""))
        answer = normalize_space(ex.get("answer", ""))
        if not context or not question or not answer:
            if progress_label and (idx == total or idx % 500 == 0):
                progress_bar(f"Preprocess S-NET {progress_label}", idx, total, len(rows))
            continue
        rows.append(
            {
                "id": ex.get("id"),
                "question": question,
                "context": context,
                "answer": answer,
                "answer_start": ex.get("answer_start"),
                "answer_end": ex.get("answer_end"),
            }
        )
        if progress_label and (idx == total or idx % 500 == 0):
            progress_bar(f"Preprocess S-NET {progress_label}", idx, total, len(rows))
    return rows


def decode_logits(logits, inv_vocab: dict[int, str]) -> list[str]:
    rows = []
    for seq in logits.argmax(dim=-1).detach().cpu().tolist():
        words = []
        for idx in seq:
            word = inv_vocab.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in {"<pad>", "<bos>"}:
                words.append(word)
        rows.append(" ".join(words))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/QA/train_data.json")
    parser.add_argument("--dev-data", default="dataset/QA/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default="models/snet")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embed-size", type=int, default=300)
    parser.add_argument("--feature-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs. Early stopping may stop sooner.")
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without dev ROUGE-L improvement.")
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
        raise SystemExit("S-NET training requires PyTorch.") from exc

    train_rows = build_rows(args.train_data, args.context_dir, args.train_limit, args.max_context_chars, progress_label="train")
    dev_rows = build_rows(args.dev_data, args.context_dir, args.dev_limit, args.max_context_chars, progress_label="dev")
    if not train_rows or not dev_rows:
        train_probe = load_examples(args.train_data, 1)
        dev_probe = load_examples(args.dev_data, 1)
        train_keys = sorted(train_probe[0].keys()) if train_probe else []
        dev_keys = sorted(dev_probe[0].keys()) if dev_probe else []
        raise SystemExit(
            "No S-NET rows were created. S-NET uses only the current sample's gold context: "
            "embedded sample text or the contexts[*].content file referenced by that sample. "
            f"First train keys={train_keys}, gold_context_found={bool(train_probe and sample_gold_context(train_probe[0], args.context_dir))}; "
            f"first dev keys={dev_keys}, gold_context_found={bool(dev_probe and sample_gold_context(dev_probe[0], args.context_dir))}."
        )
    vocab = build_vocab(train_rows + dev_rows, args.min_freq)
    if len(vocab) > args.vocab_size:
        keep = {tok: idx for tok, idx in vocab.items() if idx < 4}
        for tok, _idx in sorted(vocab.items(), key=lambda item: item[1]):
            if tok not in keep:
                keep[tok] = len(keep)
            if len(keep) >= args.vocab_size:
                break
        vocab = keep
    inv_vocab = {idx: tok for tok, idx in vocab.items()}

    class SNetDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            passage_tokens = tokenize(row["context"])[: args.max_context_tokens]
            start_flags, end_flags = token_feature_flags(
                passage_tokens,
                row["context"],
                row.get("answer_start") if isinstance(row.get("answer_start"), int) else None,
                row.get("answer_end") if isinstance(row.get("answer_end"), int) else None,
            )
            start_flags = start_flags[: args.max_context_tokens] + [0] * (args.max_context_tokens - len(start_flags))
            end_flags = end_flags[: args.max_context_tokens] + [0] * (args.max_context_tokens - len(end_flags))
            return {
                "passage_ids": torch.tensor(encode(passage_tokens, vocab, args.max_context_tokens), dtype=torch.long),
                "question_ids": torch.tensor(encode(tokenize(row["question"]), vocab, args.max_question_tokens), dtype=torch.long),
                "start_features": torch.tensor(start_flags[: args.max_context_tokens], dtype=torch.long),
                "end_features": torch.tensor(end_flags[: args.max_context_tokens], dtype=torch.long),
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

    model = SNetSynthesis(
        len(vocab),
        vocab["<pad>"],
        vocab["<bos>"],
        vocab["<eos>"],
        embed_size=args.embed_size,
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(SNetDataset(train_rows), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(SNetDataset(dev_rows), batch_size=args.batch_size, **loader_kwargs)
    best_dev_rouge = None
    bad_epochs = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context():
                output = model(**batch)
                loss = output.loss
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
        dev_predictions: list[str] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast_context():
                    loss = model(**batch).loss
                    generated = model(
                        passage_ids=batch["passage_ids"],
                        question_ids=batch["question_ids"],
                        start_features=batch["start_features"],
                        end_features=batch["end_features"],
                        max_answer_len=args.max_answer_tokens,
                    )
                dev_loss += float(loss.detach().float().cpu())
                dev_predictions.extend(decode_logits(generated.logits, inv_vocab))
        dev_loss /= max(1, len(dev_loader))
        dev_references = [row["answer"] for row in dev_rows[: len(dev_predictions)]]
        dev_rouge_l = sum(rouge_l(pred, ref) for pred, ref in zip(dev_predictions, dev_references)) / max(1, len(dev_predictions))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f} dev_rouge_l={dev_rouge_l:.4f}")
        if best_dev_rouge is None or dev_rouge_l > best_dev_rouge:
            best_dev_rouge = dev_rouge_l
            bad_epochs = 0
            torch.save(model.state_dict(), Path(args.output_dir) / "pytorch_model.bin")
            with (Path(args.output_dir) / "snet_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args) | {"vocab_size": len(vocab)}, f, ensure_ascii=False, indent=2)
            with (Path(args.output_dir) / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
        else:
            bad_epochs += 1
            print(f"dev ROUGE-L did not improve for {bad_epochs}/{args.patience} epochs")
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_dev_rouge_l={best_dev_rouge:.4f}")
                break


if __name__ == "__main__":
    main()
