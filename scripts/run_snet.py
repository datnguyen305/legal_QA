#!/usr/bin/env python3
"""Run S-NET extraction-then-synthesis inference."""

from __future__ import annotations

import argparse

from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import normalize_space
from model_architectures.snet_model import select_evidence_sentence, snet_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--output", default="outputs/snet_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("S-NET inference requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    rows = []
    for ex in load_examples(args.data, args.limit):
        question = normalize_space(ex.get("question", ""))
        context = normalize_space(ex.get("context", ""))[: args.max_context_chars]
        evidence = select_evidence_sentence(question, context).text
        encoded = tokenizer(snet_input(question, context, evidence), max_length=args.max_input_length, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**encoded, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
        rows.append({"id": ex.get("id"), "question": question, "reference": ex.get("answer", ""), "prediction": prediction, "model": "S-NET"})
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
