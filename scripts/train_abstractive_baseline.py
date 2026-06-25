#!/usr/bin/env python3
"""Train from-scratch abstractive QA baselines."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.cpg_preprocess import progress_bar, sample_gold_context
from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import normalize_space, tokenize
from evaluate_predictions import rouge_l
from train_cpg import build_vocab, encode


MODEL_CHOICES = ("dcmn_plus", "multi_style_generative", "gaqa")


def build_rows(path: str, context_dir: str, limit: int | None, max_context_chars: int, progress_label: str) -> list[dict]:
    rows = []
    examples = load_examples(path, limit)
    total = len(examples)
    print(f"Loading abstractive rows for {progress_label}: {total} examples", flush=True)
    for idx, ex in enumerate(examples, start=1):
        context = sample_gold_context(ex, context_dir)[:max_context_chars]
        question = normalize_space(ex.get("question", ""))
        answer = normalize_space(ex.get("answer", ""))
        if context and question and answer:
            rows.append({"id": ex.get("id"), "question": question, "context": context, "answer": answer, "style": 0})
        if idx == total or idx % 500 == 0:
            progress_bar(f"Preprocess abstractive {progress_label}", idx, total, len(rows))
    return rows


def make_model(name: str, vocab: dict, hidden: int, decoder_hidden: int, num_styles: int):
    from model_architectures.abstractive_baselines import DCMNPlusGenerator, GAQAGenerator, MultiStyleGenerativeRC

    args = (len(vocab), vocab["<pad>"], vocab["<bos>"], vocab["<eos>"])
    if name == "dcmn_plus":
        return DCMNPlusGenerator(*args, hidden=hidden, decoder_hidden=decoder_hidden)
    if name == "multi_style_generative":
        return MultiStyleGenerativeRC(*args, hidden=hidden, decoder_hidden=decoder_hidden, num_styles=num_styles)
    if name == "gaqa":
        return GAQAGenerator(*args, hidden=hidden, decoder_hidden=decoder_hidden)
    raise ValueError(name)


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
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--num-styles", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
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
        raise SystemExit("Abstractive baseline training requires PyTorch.") from exc

    train_rows = build_rows(args.train_data, args.context_dir, args.train_limit, args.max_context_chars, "train")
    dev_rows = build_rows(args.dev_data, args.context_dir, args.dev_limit, args.max_context_chars, "dev")
    if not train_rows or not dev_rows:
        raise SystemExit("No abstractive rows were created.")
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
                "style_ids": torch.tensor(row.get("style", 0), dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    model = make_model(args.model, vocab, args.hidden, args.decoder_hidden, args.num_styles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.startswith("cuda"), "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(QADataset(train_rows), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(QADataset(dev_rows), batch_size=args.batch_size, **loader_kwargs)
    out_dir = Path(args.output_dir or f"models/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_rouge = None
    bad_epochs = 0
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
        dev_predictions = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_context():
                    loss = model(**batch).loss
                    generated = model(
                        context_ids=batch["context_ids"],
                        question_ids=batch["question_ids"],
                        style_ids=batch["style_ids"],
                        max_answer_len=args.max_answer_tokens,
                    )
                dev_loss += float(loss.detach().float().cpu())
                dev_predictions.extend(decode_logits(generated.logits, inv_vocab))
        dev_loss /= max(1, len(dev_loader))
        dev_refs = [row["answer"] for row in dev_rows[: len(dev_predictions)]]
        dev_rouge = sum(rouge_l(pred, ref) for pred, ref in zip(dev_predictions, dev_refs)) / max(1, len(dev_predictions))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f} dev_rouge_l={dev_rouge:.4f}")
        if best_dev_rouge is None or dev_rouge > best_dev_rouge:
            best_dev_rouge = dev_rouge
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            json.dump(vars(args) | {"vocab_size": len(vocab)}, open(out_dir / "abstractive_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            json.dump(vocab, open(out_dir / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
        else:
            bad_epochs += 1
            print(f"dev ROUGE-L did not improve for {bad_epochs}/{args.patience} epochs")
            if bad_epochs >= args.patience:
                break


if __name__ == "__main__":
    main()
