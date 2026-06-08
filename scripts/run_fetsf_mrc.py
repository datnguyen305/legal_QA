#!/usr/bin/env python3
"""Run inference with the proposed FETSF-MRC model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.qa_preprocess import make_extractive_record, sentence_evidence_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/fetsf_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("FETSF-MRC inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.fetsf_model import FetsfMRC

    config_path = Path(args.model_dir) / "fetsf_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FetsfMRC(config["base_model"], max_sentences=config["max_sentences"]).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "pytorch_model.bin", map_location=device))
    model.eval()

    rows = []
    for i, ex in enumerate(load_examples(args.data, args.limit), start=1):
        record = make_extractive_record(ex, args.context_dir, args.max_context_chars, prefer_article=True)
        if record is None:
            context = ""
            prediction = ""
        else:
            sentences, evidence = sentence_evidence_labels(
                record["context"],
                record["answer_start"],
                record["answer_end"],
            )
            encoded = tokenizer(
                record["question"],
                record["context"],
                truncation="only_second",
                max_length=config["max_length"],
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            sequence_ids = encoded.sequence_ids(0)
            sentence_spans = torch.full((1, config["max_sentences"], 2), -1, dtype=torch.long)
            cursor = 0
            for j, sentence in enumerate(sentences[: config["max_sentences"]]):
                char_start = record["context"].find(sentence, cursor)
                if char_start < 0:
                    char_start = cursor
                char_end = char_start + len(sentence)
                token_indices = [
                    tok_i
                    for tok_i, (start, end) in enumerate(offsets)
                    if sequence_ids[tok_i] == 1 and end > char_start and start < char_end
                ]
                if token_indices:
                    sentence_spans[0, j, 0] = token_indices[0]
                    sentence_spans[0, j, 1] = token_indices[-1]
                cursor = char_end

            batch = {key: value.to(device) for key, value in encoded.items()}
            batch["sentence_spans"] = sentence_spans.to(device)
            with torch.no_grad():
                output = model(**batch)
            start_idx = int(output.start_logits[0].argmax().detach().cpu())
            end_idx = int(output.end_logits[0, start_idx:].argmax().detach().cpu()) + start_idx
            if sequence_ids[start_idx] != 1 or sequence_ids[end_idx] != 1:
                prediction = ""
            else:
                char_start = offsets[start_idx][0]
                char_end = offsets[end_idx][1]
                prediction = record["context"][char_start:char_end].strip()
            context = record["context"]

        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": "FETSF-MRC",
            }
        )
        if i % 50 == 0:
            print(f"Processed {i}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
