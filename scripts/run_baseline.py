#!/usr/bin/env python3
"""Run lightweight legal-QA baselines and save predictions as JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.legalqa_data import context_summary, context_text, load_examples, write_jsonl


def predict(
    example: dict,
    baseline: str,
    context_dir: str,
    max_context_chars: int | None,
    prefer_article: bool,
) -> str:
    question = (example.get("question") or "").strip()
    metadata = context_summary(example)

    if baseline == "question":
        return question
    if baseline == "metadata":
        return metadata or question
    if baseline == "context":
        return context_text(example, context_dir, max_context_chars, prefer_article) or metadata or question
    if baseline == "paper_gold":
        return example.get("answer") or ""
    raise ValueError(f"Unknown baseline: {baseline}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--output", default="outputs/baseline_predictions.jsonl")
    parser.add_argument(
        "--baseline",
        choices=["question", "metadata", "context", "paper_gold"],
        default="metadata",
        help=(
            "question copies the question; metadata answers with available legal "
            "document metadata; context copies the referenced legal passage; "
            "paper_gold is an oracle sanity check."
        ),
    )
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--no-article-slice", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.data, args.limit)
    rows = (
        {
            "id": ex.get("id"),
            "question": ex.get("question", ""),
            "reference": ex.get("answer", ""),
            "prediction": predict(
                ex,
                args.baseline,
                args.context_dir,
                args.max_context_chars,
                prefer_article=not args.no_article_slice,
            ),
            "baseline": args.baseline,
        }
        for ex in examples
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(examples)} predictions to {args.output}")


if __name__ == "__main__":
    main()
