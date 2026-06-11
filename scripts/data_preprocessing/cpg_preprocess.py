"""Curriculum preprocessing for pointer-generator long-context QA."""

from __future__ import annotations

import random
import sys
from typing import Any

from data_preprocessing.legalqa_data import context_text, load_examples
from data_preprocessing.qa_preprocess import normalize_space, tokenize


EMBEDDED_CONTEXT_FIELDS = (
    "context",
    "gold_context",
    "gold_context_text",
    "passage",
    "paragraph",
    "source_context",
    "source",
)


def progress_bar(label: str, current: int, total: int, kept: int, width: int = 30) -> None:
    total = max(1, total)
    current = min(current, total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * current / total
    sys.stderr.write(f"\r{label}: [{bar}] {current}/{total} ({percent:5.1f}%) kept={kept}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def sample_gold_context(example: dict[str, Any], context_dir: str) -> str:
    """Return only the gold context attached to the current sample.

    This does not search the context pool. It first uses context text embedded
    in the sample, then falls back to the ``contexts[*].content`` filename(s)
    referenced by that same sample.
    """
    for field in EMBEDDED_CONTEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return normalize_space(value)
        if isinstance(value, list):
            parts = [item for item in value if isinstance(item, str) and item.strip()]
            if parts:
                return normalize_space(" ".join(parts))

    contexts = example.get("contexts")
    context_values = contexts.values() if isinstance(contexts, dict) else contexts
    if isinstance(context_values, list) or hasattr(context_values, "__iter__"):
        parts: list[str] = []
        for ctx in context_values:
            if isinstance(ctx, str) and ctx.strip():
                parts.append(ctx)
            elif isinstance(ctx, dict):
                for field in EMBEDDED_CONTEXT_FIELDS:
                    value = ctx.get(field)
                    if isinstance(value, str) and value.strip():
                        parts.append(value)
                        break
        if parts:
            return normalize_space(" ".join(parts))

    return context_text(example, context_dir, None, prefer_article=False)


def chunk_tokens(tokens: list[str], chunk_size: int) -> list[list[str]]:
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)] or [[]]


def score_chunk(query_tokens: set[str], chunk: list[str]) -> float:
    if not chunk:
        return 0.0
    chunk_set = set(chunk)
    overlap = len(query_tokens & chunk_set)
    return overlap / max(1, len(query_tokens)) + overlap / max(1, len(chunk_set))


def retrieve_context(raw_context: str, query: str, chunk_size: int, max_context_tokens: int) -> str:
    tokens = tokenize(raw_context)
    query_tokens = set(tokenize(query))
    ranked = sorted(chunk_tokens(tokens, chunk_size), key=lambda chunk: score_chunk(query_tokens, chunk), reverse=True)
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
    raw_context = sample_gold_context(example, context_dir)
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
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    examples = load_examples(data_path, limit)
    total = len(examples)
    if progress_label:
        print(f"Loading CPG records for {progress_label}: {total} examples", file=sys.stderr, flush=True)
    for idx, example in enumerate(examples, start=1):
        chunk_size = rng.choice(chunk_sizes)
        query_mode = "answer" if rng.random() < easy_ratio else "question"
        record = make_cpg_record(example, context_dir, chunk_size, max_context_tokens, query_mode)
        if record is not None:
            records.append(record)
        if progress_label and (idx == total or idx % 500 == 0):
            progress_bar(f"Preprocess CPG {progress_label}", idx, total, len(records))
    return records
