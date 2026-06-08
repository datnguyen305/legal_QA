"""Data preparation for Select, Answer and Explain over legal QA contexts."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from data_preprocessing.legalqa_data import iter_context_metadata, load_context_texts
from data_preprocessing.qa_preprocess import answer_span, normalize_space, sentence_evidence_labels, sentence_split


TERM_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)


def context_refs(example: dict[str, Any]) -> list[str]:
    refs = []
    for ctx in iter_context_metadata(example):
        content = ctx.get("content")
        if content:
            refs.append(str(content))
    return refs


def load_passage(context_dir: str | Path, content: str, max_chars: int | None = None) -> str:
    path = Path(context_dir) / content
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    text = normalize_space(data.get("passage", "") if isinstance(data, dict) else "")
    return text[:max_chars] if max_chars is not None else text


def build_context_pool(examples: list[dict[str, Any]]) -> list[str]:
    seen = set()
    refs = []
    for ex in examples:
        for ref in context_refs(ex):
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def sample_candidate_refs(
    example: dict[str, Any],
    pool: list[str],
    max_docs: int,
    rng: random.Random,
) -> tuple[list[str], list[int], list[int]]:
    gold = context_refs(example)
    refs = list(dict.fromkeys(gold))
    while len(refs) < max_docs and pool:
        candidate = rng.choice(pool)
        if candidate not in refs:
            refs.append(candidate)
    refs = refs[:max_docs]
    labels = [int(ref in gold) for ref in refs]
    scores = labels[:]
    return refs, labels, scores


def answer_type_id(answer: str) -> int:
    lowered = normalize_space(answer).lower()
    if lowered in {"yes", "có", "co"}:
        return 0
    if lowered in {"no", "không", "khong"}:
        return 1
    return 2


def graph_adjacency(sentences: list[str], doc_ids: list[int], question: str, relation_count: int = 3) -> list[list[list[float]]]:
    n = len(sentences)
    adj = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(relation_count)]
    question_terms = set(TERM_RE.findall(question.lower()))
    sent_terms = [set(TERM_RE.findall(sentence.lower())) for sentence in sentences]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if doc_ids[i] == doc_ids[j]:
                adj[0][i][j] = 1.0
            if doc_ids[i] != doc_ids[j] and sent_terms[i] & question_terms and sent_terms[j] & question_terms:
                adj[1][i][j] = 1.0
            if doc_ids[i] != doc_ids[j] and sent_terms[i] & sent_terms[j]:
                adj[2][i][j] = 1.0
    return adj


def make_sae_answer_record(
    example: dict[str, Any],
    context_dir: str,
    max_context_chars: int | None,
    max_sentences: int,
) -> dict[str, Any] | None:
    contexts = load_context_texts(
        example,
        context_dir=context_dir,
        max_chars_per_context=max_context_chars,
        prefer_article=True,
    )
    doc_ids = list(range(len(contexts)))
    if not contexts:
        return None

    pieces = []
    sentence_doc_ids = []
    for doc_id, context in zip(doc_ids, contexts):
        for sentence in sentence_split(context):
            if len(sentence_doc_ids) >= max_sentences:
                break
            pieces.append(sentence)
            sentence_doc_ids.append(doc_id)
        if len(sentence_doc_ids) >= max_sentences:
            break
    context = " ".join(pieces)
    span = answer_span(context, example.get("answer", ""))
    if span is None:
        return None
    start, end = span
    sentences, support = sentence_evidence_labels(context, start, end)
    sentences = sentences[:max_sentences]
    support = support[:max_sentences]
    sentence_doc_ids = sentence_doc_ids[: len(sentences)]
    return {
        "id": example.get("id"),
        "question": example.get("question", ""),
        "context": context,
        "reference": example.get("answer", ""),
        "answer": context[start:end],
        "answer_start": start,
        "answer_end": end,
        "sentences": sentences,
        "support": support,
        "doc_ids": sentence_doc_ids,
        "adjacency": graph_adjacency(sentences, sentence_doc_ids, example.get("question", "")),
        "answer_type": answer_type_id(example.get("answer", "")),
    }
