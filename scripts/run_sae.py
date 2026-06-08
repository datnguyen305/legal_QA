#!/usr/bin/env python3
"""Run Select, Answer and Explain inference on Legal QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing.legalqa_data import load_examples, write_jsonl
from data_preprocessing.sae_preprocess import (
    build_context_pool,
    graph_adjacency,
    load_passage,
    make_sae_answer_record,
    sample_candidate_refs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--corpus-data", nargs="*", default=None)
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", default="outputs/sae_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("SAE inference requires: python3 -m pip install -r requirements-models.txt") from exc

    from model_architectures.sae_model import SaeAnswerExplain, SaeDocumentSelector

    with (Path(args.model_dir) / "sae_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    selector = SaeDocumentSelector(config["base_model"], max_docs=config["max_docs"]).to(device)
    answer_model = SaeAnswerExplain(config["base_model"], max_sentences=config["max_sentences"]).to(device)
    selector.load_state_dict(torch.load(Path(args.model_dir) / "selector.bin", map_location=device))
    answer_model.load_state_dict(torch.load(Path(args.model_dir) / "answer_explain.bin", map_location=device))
    selector.eval()
    answer_model.eval()

    examples = load_examples(args.data, args.limit)
    corpus_examples = []
    for path in args.corpus_data or [args.data]:
        corpus_examples.extend(load_examples(path))
    pool = build_context_pool(corpus_examples)

    rows = []
    for idx, ex in enumerate(examples, start=1):
        refs, _, _ = sample_candidate_refs(ex, pool, config["max_docs"], __import__("random").Random(idx))
        docs = [load_passage(args.context_dir, ref, args.max_context_chars) for ref in refs]
        encoded_docs = tokenizer(
            [ex.get("question", "")] * len(docs),
            docs,
            truncation="only_second",
            max_length=config["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            selector_out = selector(**{key: value.unsqueeze(0).to(device) for key, value in encoded_docs.items()})
        top_indices = selector_out.doc_scores[0].topk(min(config["top_k"], len(refs))).indices.detach().cpu().tolist()
        selected_docs = [docs[i] for i in top_indices if docs[i]]
        selected_refs = [refs[i] for i in top_indices if docs[i]]

        # Build the answer/explain input from selected documents. Use the training
        # preprocessor when selected docs include the annotated gold context;
        # otherwise fall back to a span-less context and let the model predict.
        record = make_sae_answer_record(ex, args.context_dir, args.max_context_chars, config["max_sentences"])
        if record is None:
            context = " ".join(selected_docs)
            sentences = context.split(". ")[: config["max_sentences"]]
            doc_ids = [0 for _ in sentences]
            adjacency = graph_adjacency(sentences, doc_ids, ex.get("question", ""))
        else:
            context = record["context"]
            sentences = record["sentences"]
            adjacency = record["adjacency"]

        encoded = tokenizer(
            ex.get("question", ""),
            context,
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
            char_start = context.find(sentence, cursor)
            if char_start < 0:
                char_start = cursor
            char_end = char_start + len(sentence)
            toks = [
                tok_i
                for tok_i, (start, end) in enumerate(offsets)
                if sequence_ids[tok_i] == 1 and end > char_start and start < char_end
            ]
            if toks:
                sentence_spans[0, j, 0] = toks[0]
                sentence_spans[0, j, 1] = toks[-1]
            cursor = char_end
        adj_tensor = torch.zeros(1, 3, config["max_sentences"], config["max_sentences"])
        for rel in range(min(3, len(adjacency))):
            for i in range(min(config["max_sentences"], len(adjacency[rel]))):
                for j in range(min(config["max_sentences"], len(adjacency[rel][i]))):
                    adj_tensor[0, rel, i, j] = adjacency[rel][i][j]
        batch = {key: value.to(device) for key, value in encoded.items()}
        batch["sentence_spans"] = sentence_spans.to(device)
        batch["adjacency"] = adj_tensor.to(device)
        with torch.no_grad():
            output = answer_model(**batch)
        answer_type = int(output.answer_type_logits[0].argmax().detach().cpu())
        if answer_type == 0:
            prediction = "Có"
        elif answer_type == 1:
            prediction = "Không"
        else:
            start_idx = int(output.start_logits[0].argmax().detach().cpu())
            end_idx = int(output.end_logits[0, start_idx:].argmax().detach().cpu()) + start_idx
            if sequence_ids[start_idx] == 1 and sequence_ids[end_idx] == 1:
                prediction = context[offsets[start_idx][0] : offsets[end_idx][1]].strip()
            else:
                prediction = ""
        support_scores = output.support_logits[0].detach().cpu()
        valid_support = []
        for sent_idx, sentence in enumerate(sentences[: config["max_sentences"]]):
            if sentence_spans[0, sent_idx, 0] >= 0:
                valid_support.append({"sentence": sentence, "score": float(support_scores[sent_idx])})
        valid_support.sort(key=lambda row: row["score"], reverse=True)
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": "SAE",
                "selected_contexts": selected_refs,
                "explanations": valid_support[:5],
            }
        )
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(examples)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
