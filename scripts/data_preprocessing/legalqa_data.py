"""Shared dataset helpers for Vietnamese legal QA splits."""

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


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _structured_document_text(data: dict[str, Any]) -> str:
    """Serialize a structured legal document while preserving its hierarchy."""
    parts: list[str] = []

    def visit(value: Any, label: str | None = None) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(f"{label}: {text}" if label else text)
            return
        if isinstance(value, (int, float)):
            parts.append(f"{label}: {value}" if label else str(value))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, label)
            return
        if not isinstance(value, dict):
            return

        number = value.get("number")
        title = value.get("title")
        if number is not None:
            parts.append(f"{label or 'Mục'} {number}")
        if isinstance(title, str) and title.strip():
            parts.append(title.strip())
        for key, item in value.items():
            if key not in {"number", "title"}:
                visit(item, key)

    for key, value in data.items():
        visit(value, key)
    return "\n".join(parts)


def _resolve_structured_path(reference: Any, context_dir: str | Path) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    base = Path(context_dir)
    reference_path = Path(reference)
    candidates = (
        reference_path,
        base.parent / reference_path,
        base.parent.parent / reference_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


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
    return window[:max_chars] if max_chars is not None else window


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
    """Load legal passages referenced by an example's context metadata.

    If a split already contains an embedded ``context`` field, that text is used
    first. This keeps preprocessing fast after extractive labels have been
    materialized.
    """
    embedded = example.get("context")
    if isinstance(embedded, str) and embedded.strip():
        text = _clean_text(embedded)
        return [text[:max_chars_per_context] if max_chars_per_context is not None else text]

    base = Path(context_dir)
    passages: list[str] = []
    for ctx in iter_context_metadata(example):
        passage: str | None = None
        structured_path = _resolve_structured_path(ctx.get("structured_path"), base)
        if structured_path is not None:
            with structured_path.open("r", encoding="utf-8") as f:
                structured_data = json.load(f)
            if isinstance(structured_data, dict):
                passage = _structured_document_text(structured_data)

        content = ctx.get("content")
        if passage is None and content:
            path = base / str(content)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                passage = data.get("passage") if isinstance(data, dict) else None
        if not isinstance(passage, str):
            continue
        passage = _clean_text(passage)
        if prefer_article:
            passage = _extract_article_window(passage, _infer_article(example, ctx), max_chars_per_context)
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
    return "\n\n".join(load_context_texts(example, context_dir, max_chars_per_context, prefer_article))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
