"""Preprocessing helpers for DynSAN / TD-SAN multi-passage QA."""

from __future__ import annotations

import random
from typing import Any

from data_preprocessing.legalqa_data import context_text, load_examples
from data_preprocessing.qa_preprocess import answer_span, normalize_space, sentence_split
from data_preprocessing.sae_preprocess import build_context_pool, context_refs, load_passage


def make_tdsan_record(
    example: dict[str, Any],
    context_dir: str = "dataset/contexts",
    max_context_chars: int | None = 12000,
) -> dict[str, Any] | None:
    context = context_text(example, context_dir, max_context_chars, prefer_article=True)
    span = answer_span(context, example.get("answer", ""))
    if span is None:
        return None
    start, end = span
    return {
        "id": example.get("id"),
        "question": normalize_space(example.get("question", "")),
        "passages": [context],
        "reference": example.get("answer", ""),
        "answer": context[start:end],
        "answer_start": start,
        "answer_end": end,
    }


def make_tdsan_multipassage_record(
    example: dict[str, Any],
    pool: list[str],
    context_dir: str,
    max_passages: int,
    max_context_chars: int | None,
    rng: random.Random,
) -> dict[str, Any] | None:
    gold = context_refs(example)
    passages = []
    for ref in gold:
        text = context_text(example, context_dir, max_context_chars, prefer_article=True)
        if text:
            passages.append(text)
            break
    while len(passages) < max_passages and pool:
        ref = rng.choice(pool)
        if ref not in gold:
            text = load_passage(context_dir, ref, max_context_chars)
            if text:
                passages.append(text)
    if not passages:
        return None
    joined = " ".join(passages)
    span = answer_span(joined, example.get("answer", ""))
    if span is None:
        return None
    start, end = span
    return {
        "id": example.get("id"),
        "question": normalize_space(example.get("question", "")),
        "passages": passages[:max_passages],
        "reference": example.get("answer", ""),
        "answer": joined[start:end],
        "answer_start": start,
        "answer_end": end,
    }


def load_tdsan_records(
    data_path: str,
    context_dir: str,
    limit: int | None,
    max_passages: int,
    max_context_chars: int | None,
    seed: int = 13,
) -> list[dict[str, Any]]:
    examples = load_examples(data_path, limit)
    pool = build_context_pool(examples)
    rng = random.Random(seed)
    records = []
    for example in examples:
        if max_passages <= 1:
            record = make_tdsan_record(example, context_dir, max_context_chars)
        else:
            record = make_tdsan_multipassage_record(
                example,
                pool,
                context_dir,
                max_passages,
                max_context_chars,
                rng,
            )
        if record is not None:
            records.append(record)
    return records
