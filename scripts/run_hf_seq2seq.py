#!/usr/bin/env python3
"""Generate answers with a Hugging Face seq2seq model."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples, make_prompt, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model id")
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--output", default="outputs/hf_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None, help="Examples: cpu, cuda, cuda:0")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--no-article-slice", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install with: "
            "python3 -m pip install -r requirements-models.txt"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    examples = load_examples(args.data, args.limit)
    rows = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = [
            make_prompt(
                ex,
                include_metadata=not args.no_metadata,
                include_context=not args.no_context,
                context_dir=args.context_dir,
                max_context_chars=args.max_context_chars,
                prefer_article=not args.no_article_slice,
            )
            for ex in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for ex, prompt, prediction in zip(batch, prompts, decoded):
            rows.append(
                {
                    "id": ex.get("id"),
                    "question": ex.get("question", ""),
                    "prompt": prompt,
                    "reference": ex.get("answer", ""),
                    "prediction": prediction.strip(),
                    "model": args.model,
                }
            )
        print(f"Generated {min(start + args.batch_size, len(examples))}/{len(examples)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
