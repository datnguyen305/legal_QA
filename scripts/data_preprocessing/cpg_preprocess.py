"""Curriculum preprocessing for pointer-generator narrative-style QA."""

from __future__ import annotations

import random
from typing import Any

from data_preprocessing.legalqa_data import context_text, load_examples
from data_preprocessing.qa_preprocess import normalize_space, tokenize


def chunk_tokens(tokens: list[str], chunk_size: int) -> list[list[str]]:
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)] or [[]]


def score_chunk(query_tokens: set[str], chunk: list[str]) -> float:
    if not chunk:
        return 0.0
    chunk_set = set(chunk)
    return len(query_tokens & chunk_set) / max(1, len(query_tokens)) + len(query_tokens & chunk_set) / max(1, len(chunk_set))


def retrieve_context(raw_context: str, query: str, chunk_size: int, max_context_tokens: int) -> str:
    tokens = tokenize(raw_context)
    query_tokens = set(tokenize(query))
    chunks = chunk_tokens(tokens, chunk_size)
    ranked = sorted(chunks, key=lambda chunk: score_chunk(query_tokens, chunk), reverse=True)
    selected: list[str] = []
    for chunk in ranked:
        selected.extend(chunk)
        if len(selected) >= max_context_tokens:
            break
    return " ".join(selected[:max_context_tokens])


def make_cpg_record(
    example: dict[str, Any],
    context_dir: str,
    chunk_size: int,
    max_context_tokens: int,
    query_mode: str,
) -> dict[str, Any] | None:
    raw_context = context_text(example, context_dir, None, prefer_article=False)
    if not raw_context:
        return None
    question = normalize_space(example.get("question", ""))
    answer = normalize_space(example.get("answer", ""))
    query = answer if query_mode == "answer" else question
    context = retrieve_context(raw_context, query, chunk_size, max_context_tokens)
    if not context or not answer:
        return None
    return {
        "id": example.get("id"),
        "question": question,
        "context": context,
        "answer": answer,
        "reference": answer,
        "chunk_size": chunk_size,
        "query_mode": query_mode,
    }


def load_cpg_records(
    data_path: str,
    context_dir: str,
    limit: int | None,
    chunk_sizes: list[int],
    max_context_tokens: int,
    easy_ratio: float,
    seed: int = 23,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for example in load_examples(data_path, limit):
        chunk_size = rng.choice(chunk_sizes)
        query_mode = "answer" if rng.random() < easy_ratio else "question"
        record = make_cpg_record(example, context_dir, chunk_size, max_context_tokens, query_mode)
        if record is not None:
            records.append(record)
    return records
