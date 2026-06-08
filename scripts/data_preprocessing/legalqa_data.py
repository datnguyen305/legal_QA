"""Shared dataset helpers for the Vietnamese legal QA splits."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def load_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load keyed-object or list-style JSON splits into a stable list."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if isinstance(value, dict):
                value = dict(value)
                value.setdefault("id", str(key))
                items.append(value)
    elif isinstance(data, list):
        items = [dict(item, id=str(i)) for i, item in enumerate(data)]
    else:
        raise ValueError(f"Unsupported dataset shape in {path}: {type(data).__name__}")

    if limit is not None:
        return items[:limit]
    return items


def iter_context_metadata(example: dict[str, Any]) -> Iterable[dict[str, Any]]:
    contexts = example.get("contexts") or {}
    if isinstance(contexts, dict):
        for value in contexts.values():
            if isinstance(value, dict):
                yield value
    elif isinstance(contexts, list):
        for value in contexts:
            if isinstance(value, dict):
                yield value


def context_summary(example: dict[str, Any]) -> str:
    """Build a compact context string from available metadata.

    The repository contains references to context JSON files, but not the files
    themselves. This summary keeps model inputs useful until full legal text is
    added.
    """
    parts: list[str] = []
    for ctx in iter_context_metadata(example):
        document = ctx.get("document")
        article = ctx.get("điều") or ctx.get("dieu")
        clause = ctx.get("khoản") or ctx.get("khoan")
        doc_type = ctx.get("Loại văn bản:")
        issued = ctx.get("Ngày ban hành:")
        fields = []
        if doc_type:
            fields.append(str(doc_type))
        if document:
            fields.append(str(document))
        if article:
            fields.append(f"Điều {article}")
        if clause:
            fields.append(f"Khoản {clause}")
        if issued:
            fields.append(f"ban hành {issued}")
        if fields:
            parts.append(", ".join(fields))
    return " | ".join(parts)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_article_window(text: str, article: Any, max_chars: int | None) -> str:
    if not article:
        return text[:max_chars] if max_chars is not None else text

    article_text = re.escape(str(article).strip())
    pattern = re.compile(rf"(?i)(?:điều|dieu)\s+{article_text}\b\.?")
    match = pattern.search(text)
    if not match:
        return text[:max_chars] if max_chars is not None else text

    start = match.start()
    next_article = re.search(r"(?i)\s(?:điều|dieu)\s+\d+[a-zA-Z]?\b\.?", text[match.end() :])
    end = match.end() + next_article.start() if next_article else len(text)
    window = text[start:end].strip()
    if max_chars is not None:
        window = window[:max_chars]
    return window


def _infer_article(example: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    article = ctx.get("điều") or ctx.get("dieu")
    if article:
        return str(article)
    for field in ("answer", "question"):
        match = re.search(r"(?i)(?:điều|dieu)\s+(\d+[a-zA-Z]?)\b", example.get(field) or "")
        if match:
            return match.group(1)
    return None


def load_context_texts(
    example: dict[str, Any],
    context_dir: str | Path = "dataset/contexts",
    max_chars_per_context: int | None = 12000,
    prefer_article: bool = True,
) -> list[str]:
    """Load legal passages referenced by an example's context metadata."""
    base = Path(context_dir)
    passages: list[str] = []
    for ctx in iter_context_metadata(example):
        content = ctx.get("content")
        if not content:
            continue
        path = base / str(content)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        passage = data.get("passage") if isinstance(data, dict) else None
        if not isinstance(passage, str):
            continue
        passage = _clean_text(passage)
        if prefer_article:
            passage = _extract_article_window(
                passage,
                _infer_article(example, ctx),
                max_chars_per_context,
            )
        elif max_chars_per_context is not None:
            passage = passage[:max_chars_per_context]
        if passage:
            passages.append(passage)
    return passages


def context_text(
    example: dict[str, Any],
    context_dir: str | Path = "dataset/contexts",
    max_chars_per_context: int | None = 12000,
    prefer_article: bool = True,
) -> str:
    return "\n\n".join(
        load_context_texts(example, context_dir, max_chars_per_context, prefer_article)
    )


def make_prompt(
    example: dict[str, Any],
    include_metadata: bool = True,
    include_context: bool = True,
    context_dir: str | Path = "dataset/contexts",
    max_context_chars: int | None = 12000,
    prefer_article: bool = True,
) -> str:
    question = (example.get("question") or "").strip()
    blocks = [f"Câu hỏi: {question}"]

    if include_context:
        passage = context_text(example, context_dir, max_context_chars, prefer_article)
        if passage:
            blocks.append(f"Văn bản pháp luật liên quan:\n{passage}")

    if not include_metadata:
        blocks.append("Trả lời:")
        return "\n\n".join(blocks)

    metadata = context_summary(example)
    if metadata:
        blocks.append(f"Thông tin văn bản liên quan: {metadata}")
    blocks.append("Trả lời:")
    return "\n\n".join(blocks)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
