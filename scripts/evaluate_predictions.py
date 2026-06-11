#!/usr/bin/env python3
"""Evaluate generated Legal QA answers with BLEU@4, ROUGE-L, METEOR, and BERTScore."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu4(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    precisions = []
    for n in range(1, 5):
        p_counts = ngrams(pred, n)
        r_counts = ngrams(ref, n)
        overlap = sum(min(count, r_counts[gram]) for gram, count in p_counts.items())
        total = sum(p_counts.values())
        # Chen-Cherry style smoothing for sentence-level BLEU.
        precisions.append((overlap + 1.0) / (total + 1.0))
    bp = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(pred)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / 4)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def meteor(prediction: str, reference: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    ref_counts = Counter(ref)
    used = Counter()
    matches = []
    for i, tok in enumerate(pred):
        if used[tok] < ref_counts[tok]:
            used[tok] += 1
            matches.append(i)
    matched = len(matches)
    if matched == 0:
        return 0.0
    precision = matched / len(pred)
    recall = matched / len(ref)
    fmean = (10 * precision * recall) / (recall + 9 * precision) if precision and recall else 0.0
    chunks = 1
    for left, right in zip(matches, matches[1:]):
        if right != left + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matched) ** 3
    return fmean * (1 - penalty)


def read_predictions(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bertscore_scores(predictions: list[str], references: list[str], model_type: str, batch_size: int, device: str | None) -> tuple[list[float], list[float], list[float]]:
    try:
        from bert_score import score
    except ImportError as exc:
        raise SystemExit("BERTScore evaluation requires: python3 -m pip install bert-score") from exc
    kwargs = {"model_type": model_type, "batch_size": batch_size, "verbose": True, "rescale_with_baseline": False}
    if device:
        kwargs["device"] = device
    precision, recall, f1 = score(predictions, references, **kwargs)
    return precision.tolist(), recall.tolist(), f1.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL with prediction and reference fields")
    parser.add_argument("--output", default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-device", default=None)
    args = parser.parse_args()

    rows = read_predictions(args.predictions)
    references = [row.get("reference", row.get("answer", "")) for row in rows]
    predictions = [row.get("prediction", "") for row in rows]
    bertscore = bertscore_scores(predictions, references, args.bertscore_model, args.bertscore_batch_size, args.bertscore_device) if args.bertscore else None
    detailed = []
    for i, (row, pred, ref) in enumerate(zip(rows, predictions, references)):
        item = {"id": row.get("id"), "bleu_4": bleu4(pred, ref), "rouge_l": rouge_l(pred, ref), "meteor": meteor(pred, ref)}
        if bertscore is not None:
            p, r, f1 = bertscore
            item["bertscore_precision"] = p[i]
            item["bertscore_recall"] = r[i]
            item["bertscore_f1"] = f1[i]
        detailed.append(item)
    summary = {
        "count": len(rows),
        "bleu_4": sum(x["bleu_4"] for x in detailed) / len(detailed) if detailed else 0.0,
        "rouge_l": sum(x["rouge_l"] for x in detailed) / len(detailed) if detailed else 0.0,
        "meteor": sum(x["meteor"] for x in detailed) / len(detailed) if detailed else 0.0,
    }
    if bertscore is not None:
        summary["bertscore_model"] = args.bertscore_model
        summary["bertscore_precision"] = sum(x["bertscore_precision"] for x in detailed) / len(detailed) if detailed else 0.0
        summary["bertscore_recall"] = sum(x["bertscore_recall"] for x in detailed) / len(detailed) if detailed else 0.0
        summary["bertscore_f1"] = sum(x["bertscore_f1"] for x in detailed) / len(detailed) if detailed else 0.0
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "examples": detailed}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
