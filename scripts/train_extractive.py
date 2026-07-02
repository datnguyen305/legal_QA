#!/usr/bin/env python3
"""Train extractive paper models: QANet, Cross-Passage, Deep Cascade, TD-SAN."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from data_preprocessing.cpg_preprocess import cache_key, file_fingerprint, progress_bar, sample_gold_context
from data_preprocessing.legalqa_data import load_examples
from data_preprocessing.qa_preprocess import TOKEN_RE, make_extractive_record, normalize_space, tokenize
from evaluate_predictions import rouge_l
from train_cpg import SPECIALS, encode


MODEL_CHOICES = ("qanet", "cross_passage", "deep_cascade", "td_san")


def batch_progress_bar(label: str, current: int, total: int, loss: float | None = None, width: int = 30) -> None:
    total = max(1, total)
    current = min(current, total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * current / total
    suffix = f" loss={loss:.4f}" if loss is not None else ""
    sys.stderr.write(f"\r{label}: [{bar}] {current}/{total} ({percent:5.1f}%){suffix}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def build_vocab(records: list[dict], min_freq: int) -> dict[str, int]:
    counter = Counter()
    for row in records:
        counter.update(row["question_tokens"])
        for passage in row["passage_tokens"]:
            counter.update(passage)
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def find_subsequence(tokens: list[str], needle: list[str]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(tokens):
        return None
    for i in range(len(tokens) - len(needle) + 1):
        if tokens[i : i + len(needle)] == needle:
            return i, i + len(needle) - 1
    return None


def token_span_from_chars(text: str, start: int, end: int) -> tuple[int, int] | None:
    token_indexes = []
    for idx, match in enumerate(TOKEN_RE.finditer((text or "").lower())):
        if match.end() > start and match.start() < end:
            token_indexes.append(idx)
    if not token_indexes:
        return None
    return token_indexes[0], token_indexes[-1]


def best_overlap_span(context_tokens: list[str], answer_tokens: list[str]) -> tuple[int, int] | None:
    """Weak span label for abstractive references that do not exactly occur."""
    content = [tok for tok in answer_tokens if len(tok) > 1]
    answer_set = set(content or answer_tokens)
    if not context_tokens or not answer_set:
        return None
    max_len = min(len(context_tokens), max(8, min(len(answer_tokens), 160)))
    best_start = 0
    best_end = min(len(context_tokens), max_len) - 1
    best_score = -1.0
    for start in range(len(context_tokens)):
        counts: dict[str, int] = {}
        overlap = 0
        stop = min(len(context_tokens), start + max_len)
        for end in range(start, stop):
            tok = context_tokens[end]
            if tok in answer_set:
                counts[tok] = counts.get(tok, 0) + 1
                if counts[tok] == 1:
                    overlap += 1
            length = end - start + 1
            score = overlap / max(1, len(answer_set)) + overlap / max(1, length)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
    if best_score <= 0:
        return None
    return best_start, best_end


def split_passages(tokens: list[str], max_passages: int, passage_len: int) -> list[list[str]]:
    passages = [tokens[i : i + passage_len] for i in range(0, len(tokens), passage_len)]
    return passages[:max_passages] or [[]]


def extractive_cache_payload(
    data_path: str,
    context_dir: str,
    limit: int | None,
    max_context_chars: int,
    max_passages: int,
    passage_len: int,
) -> dict:
    return {
        "kind": "extractive_records",
        "version": 3,
        "data": file_fingerprint(data_path),
        "context_dir": str(Path(context_dir).resolve()),
        "limit": limit,
        "max_context_chars": max_context_chars,
        "max_passages": max_passages,
        "passage_len": passage_len,
    }


def extractive_cache_path(cache_dir: str, label: str, payload: dict) -> Path:
    return Path(cache_dir) / f"{label}_{cache_key(payload)}.json"


def build_records(
    path: str,
    context_dir: str,
    limit: int | None,
    max_context_chars: int,
    max_passages: int,
    passage_len: int,
    progress_label: str | None = None,
) -> list[dict]:
    examples = load_examples(path, limit)
    rows = []
    if progress_label:
        print(f"Loading extractive records for {progress_label}: {len(examples)} examples", flush=True)
    for idx, ex in enumerate(examples, start=1):
        record = make_extractive_record(ex, context_dir, max_context_chars)
        question = normalize_space(ex.get("question", ""))
        answer = normalize_space(ex.get("answer", ""))
        context = record["context"] if record is not None else sample_gold_context(ex, context_dir)[:max_context_chars]
        context_tokens = tokenize(context)
        answer_tokens = tokenize(record["answer"] if record is not None else answer)
        span = (
            token_span_from_chars(context, record["answer_start"], record["answer_end"])
            if record is not None
            else find_subsequence(context_tokens, answer_tokens)
        )
        if span is None:
            span = find_subsequence(context_tokens, answer_tokens)
        if span is None:
            span = best_overlap_span(context_tokens, tokenize(answer))
        if question and answer and span is not None:
            passages = split_passages(context_tokens, max_passages, passage_len)
            start, end = span
            if end < max_passages * passage_len:
                extractive_answer = " ".join(context_tokens[start : end + 1])
                rows.append(
                    {
                        "id": ex.get("id"),
                        "question": question,
                        "answer": extractive_answer,
                        "reference": extractive_answer,
                        "abstractive_reference": answer,
                        "question_tokens": tokenize(question),
                        "passage_tokens": passages,
                        "start": start,
                        "end": end,
                    }
                )
        if progress_label and (idx == len(examples) or idx % 500 == 0):
            progress_bar(f"Preprocess extractive {progress_label}", idx, len(examples), len(rows))
    return rows


def load_or_build_records(
    path: str,
    context_dir: str,
    limit: int | None,
    max_context_chars: int,
    max_passages: int,
    passage_len: int,
    progress_label: str | None = None,
    cache_dir: str = "cache/extractive",
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> list[dict]:
    label = (progress_label or "records").replace(" ", "_")
    payload = extractive_cache_payload(path, context_dir, limit, max_context_chars, max_passages, passage_len)
    cache_path = extractive_cache_path(cache_dir, label, payload)
    if use_cache and cache_path.exists() and not rebuild_cache:
        print(f"Loading cached extractive records from {cache_path}", file=sys.stderr, flush=True)
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    records = build_records(path, context_dir, limit, max_context_chars, max_passages, passage_len, progress_label)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        tmp_path.replace(cache_path)
        print(f"Saved cached extractive records to {cache_path}", file=sys.stderr, flush=True)
    return records


def make_model(name: str, config: dict, vocab: dict, device: str):
    if name == "qanet":
        from model_architectures.qanet_model import QANet

        return QANet(len(vocab), vocab["<pad>"], hidden=config["hidden"], heads=config["heads"], dropout=config["dropout"]).to(device)
    if name == "cross_passage":
        from model_architectures.cross_passage_model import CrossPassageAnswerVerification

        return CrossPassageAnswerVerification(len(vocab), vocab["<pad>"], hidden=config["hidden"], dropout=config["dropout"]).to(device)
    if name == "deep_cascade":
        from model_architectures.deep_cascade_model import DeepCascadeReader

        return DeepCascadeReader(len(vocab), vocab["<pad>"], hidden=config["hidden"], dropout=config["dropout"]).to(device)
    if name == "td_san":
        from model_architectures.td_san_model import TDSANReader

        return TDSANReader(
            len(vocab),
            vocab["<pad>"],
            hidden=config["hidden"],
            heads=config["heads"],
            top_k=config["top_k"],
            cross_layers=config["cross_layers"],
            max_passages=config["max_passages"],
            dropout=config["dropout"],
        ).to(device)
    raise ValueError(name)


def decode_spans(start_logits, end_logits, batch_rows: list[dict]) -> list[str]:
    predictions = []
    starts = start_logits.argmax(dim=-1).detach().cpu().tolist()
    ends = end_logits.argmax(dim=-1).detach().cpu().tolist()
    for row, start, end in zip(batch_rows, starts, ends):
        flat_tokens = [tok for passage in row["passage_tokens"] for tok in passage]
        if end < start:
            end = start
        end = min(end, len(flat_tokens) - 1)
        predictions.append(" ".join(flat_tokens[start : end + 1]))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--train-data", default="dataset/QA/train_data.json")
    parser.add_argument("--dev-data", default="dataset/QA/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="cache/extractive")
    parser.add_argument("--no-disk-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-passages", type=int, default=6)
    parser.add_argument("--passage-len", type=int, default=256)
    parser.add_argument("--max-question-tokens", type=int, default=64)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
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
        raise SystemExit("Extractive training requires PyTorch.") from exc

    use_cache = not args.no_disk_cache
    train_rows = load_or_build_records(
        args.train_data,
        args.context_dir,
        args.train_limit,
        args.max_context_chars,
        args.max_passages,
        args.passage_len,
        "train",
        args.cache_dir,
        use_cache,
        args.rebuild_cache,
    )
    dev_rows = load_or_build_records(
        args.dev_data,
        args.context_dir,
        args.dev_limit,
        args.max_context_chars,
        args.max_passages,
        args.passage_len,
        "dev",
        args.cache_dir,
        use_cache,
        args.rebuild_cache,
    )
    if not train_rows or not dev_rows:
        raise SystemExit("No extractive records were created. These models require the gold answer text to appear in the selected context.")
    vocab = build_vocab(train_rows + dev_rows, args.min_freq)

    class SpanDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            passages = row["passage_tokens"][: args.max_passages]
            while len(passages) < args.max_passages:
                passages.append([])
            passage_ids = [encode(p, vocab, args.passage_len) for p in passages]
            flat_len = args.max_passages * args.passage_len
            content = [0.0] * flat_len
            for pos in range(row["start"], row["end"] + 1):
                if pos < flat_len:
                    content[pos] = 1.0
            passage_index = min(args.max_passages - 1, row["start"] // args.passage_len)
            labels = [0.0] * args.max_passages
            labels[passage_index] = 1.0
            return {
                "passage_ids": torch.tensor(passage_ids, dtype=torch.long),
                "context_ids": torch.tensor([tok for p in passage_ids for tok in p], dtype=torch.long),
                "question_ids": torch.tensor(encode(row["question_tokens"], vocab, args.max_question_tokens), dtype=torch.long),
                "start_positions": torch.tensor(row["start"], dtype=torch.long),
                "end_positions": torch.tensor(row["end"], dtype=torch.long),
                "content_labels": torch.tensor(content, dtype=torch.float),
                "document_labels": torch.tensor(labels, dtype=torch.float),
                "paragraph_labels": torch.tensor(labels, dtype=torch.float),
                "row_index": torch.tensor(idx, dtype=torch.long),
            }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    def model_batch(batch: dict) -> dict:
        batch = {k: v.to(device) for k, v in batch.items()}
        if args.model == "qanet":
            keep = {"context_ids", "question_ids", "start_positions", "end_positions"}
        elif args.model == "cross_passage":
            keep = {"passage_ids", "question_ids", "context_ids", "start_positions", "end_positions", "content_labels"}
        elif args.model == "deep_cascade":
            keep = {
                "passage_ids",
                "question_ids",
                "context_ids",
                "start_positions",
                "end_positions",
                "document_labels",
                "paragraph_labels",
            }
        elif args.model == "td_san":
            keep = {"passage_ids", "question_ids", "context_ids", "start_positions", "end_positions"}
        else:
            keep = set(batch)
        return {key: value for key, value in batch.items() if key in keep}

    config = vars(args) | {"vocab_size": len(vocab)}
    model = make_model(args.model, config, vocab, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.startswith("cuda"), "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(SpanDataset(train_rows), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(SpanDataset(dev_rows), batch_size=args.batch_size, **loader_kwargs)
    out_dir = Path(args.output_dir or f"models/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_rouge = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            row_index = batch.pop("row_index")
            del row_index
            batch = model_batch(batch)
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
            batch_progress_bar(f"Train {args.model} epoch {epoch}", step, len(train_loader), total / step)
        model.eval()
        dev_loss = 0.0
        predictions: list[str] = []
        references: list[str] = []
        with torch.no_grad():
            for step, batch in enumerate(dev_loader, start=1):
                idxs = batch.pop("row_index").tolist()
                batch_rows = [dev_rows[i] for i in idxs]
                batch = model_batch(batch)
                with autocast_context():
                    out = model(**batch)
                dev_loss += float(out.loss.detach().float().cpu())
                predictions.extend(decode_spans(out.start_logits, out.end_logits, batch_rows))
                references.extend(row["answer"] for row in batch_rows)
                batch_progress_bar(f"Dev {args.model} epoch {epoch}", step, len(dev_loader), dev_loss / step)
        dev_loss /= max(1, len(dev_loader))
        dev_rouge_l = sum(rouge_l(pred, ref) for pred, ref in zip(predictions, references)) / max(1, len(predictions))
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} dev_loss={dev_loss:.4f} dev_rouge_l={dev_rouge_l:.4f}")
        if best_dev_rouge is None or dev_rouge_l > best_dev_rouge:
            best_dev_rouge = dev_rouge_l
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            json.dump(config, open(out_dir / "extractive_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            json.dump(vocab, open(out_dir / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)
        else:
            bad_epochs += 1
            print(f"dev ROUGE-L did not improve for {bad_epochs}/{args.patience} epochs")
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_dev_rouge_l={best_dev_rouge:.4f}")
                break


if __name__ == "__main__":
    main()
