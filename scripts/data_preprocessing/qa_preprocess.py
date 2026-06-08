"""Preprocessing helpers for extractive legal QA models."""

from __future__ import annotations

import re
from typing import Any

from data_preprocessing.legalqa_data import context_text, load_context_texts


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def sentence_split(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(normalize_space(text))
    return [part.strip() for part in parts if part.strip()]


def answer_span(context: str, answer: str) -> tuple[int, int] | None:
    """Find a usable extractive answer span inside context.

    Dataset answers often include explanatory prose around quoted legal text.
    We first try the full answer, then progressively try longer answer
    sentences and clauses.
    """
    context_norm = normalize_space(context)
    answer_norm = normalize_space(answer)
    if not context_norm or not answer_norm:
        return None

    candidates = [answer_norm]
    candidates.extend(sentence_split(answer_norm))
    candidates.extend(part.strip() for part in re.split(r"[\n;:。.!?]", answer_norm) if part.strip())
    candidates = sorted(set(candidates), key=len, reverse=True)

    lower_context = context_norm.lower()
    for candidate in candidates:
        if len(candidate) < 12:
            continue
        start = lower_context.find(candidate.lower())
        if start >= 0:
            return start, start + len(candidate)
    return None


def make_extractive_record(
    example: dict[str, Any],
    context_dir: str = "dataset/contexts",
    max_context_chars: int | None = 12000,
    prefer_article: bool = True,
) -> dict[str, Any] | None:
    context = context_text(example, context_dir, max_context_chars, prefer_article)
    answer = example.get("answer", "")
    span = answer_span(context, answer)
    if span is None:
        return None
    start, end = span
    return {
        "id": example.get("id"),
        "question": example.get("question", ""),
        "context": context,
        "answer": context[start:end],
        "answer_start": start,
        "answer_end": end,
        "reference": answer,
    }


def sentence_evidence_labels(context: str, answer_start: int, answer_end: int) -> tuple[list[str], list[int]]:
    sentences = sentence_split(context)
    labels: list[int] = []
    cursor = 0
    for sentence in sentences:
        idx = context.find(sentence, cursor)
        if idx < 0:
            idx = cursor
        sent_start = idx
        sent_end = idx + len(sentence)
        labels.append(int(sent_start <= answer_end and sent_end >= answer_start))
        cursor = sent_end
    return sentences, labels


def gold_contexts(example: dict[str, Any], context_dir: str, max_context_chars: int | None) -> list[str]:
    return load_context_texts(
        example,
        context_dir=context_dir,
        max_chars_per_context=max_context_chars,
        prefer_article=True,
    )
