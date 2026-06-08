"""Preprocessing for A Deep Cascade Model for Multi-Document MRC."""

from __future__ import annotations

import random
import re
from typing import Any

from data_preprocessing.legalqa_data import context_text, load_examples
from data_preprocessing.qa_preprocess import answer_span, normalize_space, sentence_split, tokenize
from data_preprocessing.sae_preprocess import build_context_pool, context_refs, load_passage


def text_features(question: str, text: str, position: int = 0, total: int = 1) -> list[float]:
    q = set(tokenize(question))
    t = tokenize(text)
    ts = set(t)
    overlap = len(q & ts)
    return [
        overlap / max(1, len(q)),
        overlap / max(1, len(ts)),
        len(t) / 1000.0,
        1.0 if position == 0 else 0.0,
        1.0 if position == total - 1 else 0.0,
    ]


def paragraph_split(text: str, max_words: int = 220) -> list[str]:
    sentences = sentence_split(text)
    paragraphs: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = tokenize(sentence)
        if current and count + len(words) > max_words:
            paragraphs.append(" ".join(current))
            current = []
            count = 0
        current.append(sentence)
        count += len(words)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs or [normalize_space(text)]


def make_candidate_docs(
    example: dict[str, Any],
    pool: list[str],
    context_dir: str,
    max_docs: int,
    max_context_chars: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    gold_refs = context_refs(example)
    docs: list[dict[str, Any]] = []
    gold_text = context_text(example, context_dir, max_context_chars, prefer_article=True)
    if gold_text:
        docs.append({"ref": gold_refs[0] if gold_refs else "gold", "text": gold_text, "label": 1})
    while len(docs) < max_docs and pool:
        ref = rng.choice(pool)
        if ref in gold_refs or any(doc["ref"] == ref for doc in docs):
            continue
        text = load_passage(context_dir, ref, max_context_chars)
        if text:
            docs.append({"ref": ref, "text": text, "label": 0})
    return docs[:max_docs]


def make_deep_cascade_record(
    example: dict[str, Any],
    pool: list[str],
    context_dir: str,
    max_docs: int,
    max_paragraphs: int,
    max_context_chars: int | None,
    rng: random.Random,
) -> dict[str, Any] | None:
    docs = make_candidate_docs(example, pool, context_dir, max_docs, max_context_chars, rng)
    if not docs:
        return None
    selected_docs = []
    context_parts = []
    paragraph_meta = []
    cursor = 0
    for doc_idx, doc in enumerate(docs):
        paragraphs = paragraph_split(doc["text"])[:max_paragraphs]
        p_rows = []
        for para_idx, para in enumerate(paragraphs):
            start = cursor
            context_parts.append(para)
            cursor += len(para) + 1
            end = start + len(para)
            p_rows.append({"text": para, "start": start, "end": end, "label": 0})
            paragraph_meta.append((doc_idx, para_idx, start, end))
        selected_docs.append({**doc, "paragraphs": p_rows})
    context = " ".join(context_parts)
    span = answer_span(context, example.get("answer", ""))
    if span is None:
        return None
    answer_start, answer_end = span
    doc_labels = []
    para_labels = []
    for doc_idx, doc in enumerate(selected_docs):
        doc_label = 0
        row = []
        for para in doc["paragraphs"]:
            label = int(para["start"] <= answer_end and para["end"] >= answer_start)
            para["label"] = label
            row.append(label)
            doc_label = max(doc_label, label)
        doc_labels.append(max(int(doc["label"]), doc_label))
        para_labels.append(row + [0] * (max_paragraphs - len(row)))
    return {
        "id": example.get("id"),
        "question": normalize_space(example.get("question", "")),
        "context": context,
        "reference": example.get("answer", ""),
        "answer": context[answer_start:answer_end],
        "answer_start": answer_start,
        "answer_end": answer_end,
        "docs": selected_docs,
        "doc_labels": doc_labels + [0] * (max_docs - len(doc_labels)),
        "para_labels": para_labels + [[0] * max_paragraphs for _ in range(max_docs - len(para_labels))],
        "doc_features": [
            text_features(example.get("question", ""), doc["text"], i, len(selected_docs))
            for i, doc in enumerate(selected_docs)
        ]
        + [[0.0] * 5 for _ in range(max_docs - len(selected_docs))],
        "para_features": [
            [
                text_features(example.get("question", ""), para["text"], j, len(doc["paragraphs"]))
                for j, para in enumerate(doc["paragraphs"])
            ]
            + [[0.0] * 5 for _ in range(max_paragraphs - len(doc["paragraphs"]))]
            for doc in selected_docs
        ]
        + [[[0.0] * 5 for _ in range(max_paragraphs)] for _ in range(max_docs - len(selected_docs))],
    }


def load_deep_cascade_records(
    data_path: str,
    context_dir: str,
    limit: int | None,
    max_docs: int,
    max_paragraphs: int,
    max_context_chars: int | None,
    seed: int = 17,
) -> list[dict[str, Any]]:
    examples = load_examples(data_path, limit)
    pool = build_context_pool(examples)
    rng = random.Random(seed)
    records = []
    for ex in examples:
        record = make_deep_cascade_record(ex, pool, context_dir, max_docs, max_paragraphs, max_context_chars, rng)
        if record is not None:
            records.append(record)
    return records
