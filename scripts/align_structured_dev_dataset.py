#!/usr/bin/env python3
"""Attach matching structured-document references to IR dataset splits."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any


DATASET_DIR = Path("dataset/structured-single-hop-IR")
SPLITS = ("train_data.json", "dev_data.json", "test_data.json")
CONTEXT_DIR = Path("dataset/contexts")
STRUCTURED_DIR = DATASET_DIR / "structured_data"
STRUCTURED_KEYS = ("Header", "Phần", "Chương", "Mục", "Tiểu Mục", "Điều")


def url_slug(link: str) -> str:
    slug = urllib.parse.unquote(urllib.parse.urlparse(link or "").path.rstrip("/").rsplit("/", 1)[-1])
    return slug[:-5] if slug.lower().endswith(".aspx") else slug


def load_context_key(ctx: dict[str, Any]) -> str:
    content = ctx.get("content")
    if content:
        context_path = CONTEXT_DIR / str(content)
        if context_path.is_file():
            with context_path.open("r", encoding="utf-8") as f:
                context_data = json.load(f)
            if isinstance(context_data, dict):
                name = context_data.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
                context_link = context_data.get("link")
                if isinstance(context_link, str) and context_link.strip():
                    return url_slug(context_link)
    return url_slug(str(ctx.get("link", "")))


def update_split(dataset_path: Path, structured_files: dict[str, Path]) -> tuple[int, int]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.values() if isinstance(data, dict) else data
    updated_contexts = 0
    seen_structured: set[str] = set()
    for row in rows:
        contexts = row.get("contexts") or {}
        values = contexts.values() if isinstance(contexts, dict) else contexts
        for ctx in values:
            if not isinstance(ctx, dict):
                continue
            key = load_context_key(ctx)
            structured_path = structured_files.get(key)
            if structured_path is None:
                raise FileNotFoundError(f"No structured document for context {ctx.get('content')}: {key}")
            relative_path = structured_path.resolve().relative_to(DATASET_DIR.parent.resolve())
            ctx["structured_path"] = str(relative_path)
            ctx["structured_type"] = structured_path.parent.name
            ctx["structured_schema"] = list(STRUCTURED_KEYS)
            updated_contexts += 1
            seen_structured.add(str(relative_path))

    temp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(dataset_path)
    return updated_contexts, len(seen_structured)


def main() -> None:
    structured_files = {path.stem: path for path in STRUCTURED_DIR.glob("*/*.json")}
    for split in SPLITS:
        updated_contexts, unique_structured = update_split(DATASET_DIR / split, structured_files)
        print(f"{split}: updated contexts={updated_contexts}, unique structured documents={unique_structured}")


if __name__ == "__main__":
    main()
