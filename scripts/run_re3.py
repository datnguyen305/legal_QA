#!/usr/bin/env python3
"""Run RE3QA inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.re3_preprocess import make_re3_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/re3.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("RE3QA inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.re3_model import Re3QA

    config = json.load(open(Path(args.model_dir) / "re3_config.json", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Re3QA(config["base_model"], early_layer=config["early_layer"], max_candidates=config["max_candidates"]).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()

    rows = []
    for ex in load_examples(args.data, args.limit):
        best = None
        for seg in make_re3_segments(ex, args.context_dir, config["window_tokens"], config["stride"], args.max_context_chars):
            enc = tokenizer(
                seg["question"],
                seg["segment"],
                truncation="only_second",
                max_length=config["max_length"],
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            sequence_ids = enc.sequence_ids(0)
            batch = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**batch)
            retrieve_score = float(out.retrieve_logits[0, 1].detach().cpu())
            start_logits = out.start_logits[0].detach().cpu()
            end_logits = out.end_logits[0].detach().cpu()
            rerank_score = float(out.rerank_logits[0, 0].detach().cpu())
            start = int(start_logits.argmax())
            end = int(end_logits[start:].argmax()) + start
            if sequence_ids[start] != 1 or sequence_ids[end] != 1:
                continue
            read_score = float(start_logits[start] + end_logits[end])
            score = 1.4 * retrieve_score + read_score + 1.4 * rerank_score
            pred = seg["segment"][offsets[start][0] : offsets[end][1]].strip()
            if best is None or score > best[0]:
                best = (score, pred)
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": best[1] if best else "",
                "model": "RE3QA",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
