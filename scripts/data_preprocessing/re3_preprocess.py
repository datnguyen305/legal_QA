"""Preprocessing for RE3QA: retrieve, read, rerank."""

from __future__ import annotations

from typing import Any

from data_preprocessing.legalqa_data import context_text, load_examples
from data_preprocessing.qa_preprocess import answer_span, normalize_space, tokenize


def sliding_segments(tokens: list[str], window: int, stride: int) -> list[tuple[list[str], int]]:
    if len(tokens) <= window:
        return [(tokens, 0)]
    out = []
    for start in range(0, max(1, len(tokens) - window + 1), stride):
        out.append((tokens[start : start + window], start))
    if out and out[-1][1] + len(out[-1][0]) < len(tokens):
        start = max(0, len(tokens) - window)
        out.append((tokens[start:], start))
    return out


def make_re3_segments(
    example: dict[str, Any],
    context_dir: str,
    window_tokens: int,
    stride: int,
    max_context_chars: int | None,
) -> list[dict[str, Any]]:
    context = context_text(example, context_dir, max_context_chars, prefer_article=True)
    if not context:
        return []
    answer = normalize_space(example.get("answer", ""))
    question = normalize_space(example.get("question", ""))
    context_tokens = tokenize(context)
    rows = []
    for seg_tokens, token_offset in sliding_segments(context_tokens, window_tokens, stride):
        seg_text = " ".join(seg_tokens)
        span = answer_span(seg_text, answer)
        rows.append(
            {
                "id": example.get("id"),
                "question": question,
                "segment": seg_text,
                "reference": answer,
                "has_answer": int(span is not None),
                "answer_start": span[0] if span else None,
                "answer_end": span[1] if span else None,
                "token_offset": token_offset,
            }
        )
    return rows


def load_re3_segments(
    data_path: str,
    context_dir: str,
    limit: int | None,
    window_tokens: int,
    stride: int,
    max_context_chars: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for example in load_examples(data_path, limit):
        rows.extend(make_re3_segments(example, context_dir, window_tokens, stride, max_context_chars))
    return rows
