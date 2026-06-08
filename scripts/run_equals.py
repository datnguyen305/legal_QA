#!/usr/bin/env python3
"""Run the EQUALS proposed retrieval + MRC pipeline.

The paper's proposed system retrieves relevant law articles with BM25 or
Sentence-BERT, then extracts the exact answer with a BERT-style MRC model.
This script implements that pipeline for the local Legal QA dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_preprocessing.legalqa_data import iter_context_metadata, load_examples, write_jsonl
from data_preprocessing.qa_preprocess import gold_contexts, normalize_space, tokenize


class BM25Retriever:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc["text"]) for doc in docs]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self.tokens:
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log(1 + (len(docs) - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.lengths[i] or 1
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]


class SentenceBertRetriever:
    def __init__(self, docs: list[dict[str, Any]], model_name: str, batch_size: int, device: str | None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Sentence-BERT retrieval requires: python3 -m pip install -r requirements-models.txt"
            ) from exc

        self.docs = docs
        self.torch = torch
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode(
            [doc["text"] for doc in docs],
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = query_embedding @ self.embeddings.T
        indices = self.torch.topk(scores[0], k=min(top_k, len(self.docs))).indices.tolist()
        return [self.docs[i] for i in indices]


def referenced_context_files(examples: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for example in examples:
        for ctx in iter_context_metadata(example):
            content = ctx.get("content")
            if content and content not in seen:
                seen.add(str(content))
                ordered.append(str(content))
    return ordered


def build_corpus(
    examples: list[dict[str, Any]],
    context_dir: str,
    max_context_chars: int | None,
) -> list[dict[str, Any]]:
    docs = []
    for content in referenced_context_files(examples):
        path = Path(context_dir) / content
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        text = normalize_space(data.get("passage", "") if isinstance(data, dict) else "")
        if max_context_chars is not None:
            text = text[:max_context_chars]
        if text:
            docs.append({"id": data.get("id", content), "content": content, "text": text})
    return docs


def extract_answer(
    question: str,
    context: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_length: int,
    max_answer_length: int,
) -> str:
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    start_logits = outputs.start_logits[0].detach().cpu()
    end_logits = outputs.end_logits[0].detach().cpu()

    best_score = None
    best_span = (0, 0)
    for start in torch.topk(start_logits, k=min(20, len(start_logits))).indices.tolist():
        if sequence_ids[start] != 1:
            continue
        for end in torch.topk(end_logits, k=min(20, len(end_logits))).indices.tolist():
            if sequence_ids[end] != 1 or end < start or end - start + 1 > max_answer_length:
                continue
            score = float(start_logits[start] + end_logits[end])
            if best_score is None or score > best_score:
                best_score = score
                best_span = (offsets[start][0], offsets[end][1])
    return context[best_span[0] : best_span[1]].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/test_data.json")
    parser.add_argument("--corpus-data", nargs="*", default=None)
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--retriever", choices=["gold", "bm25", "sbert"], default="gold")
    parser.add_argument("--qa-model", required=True, help="Fine-tuned extractive QA model path or HF id")
    parser.add_argument("--sbert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-answer-length", type=int, default=261)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/equals_predictions.jsonl")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("EQUALS MRC requires: python3 -m pip install -r requirements-models.txt") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.qa_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model).to(device)
    model.eval()

    examples = load_examples(args.data, args.limit)
    retriever = None
    if args.retriever != "gold":
        corpus_examples = []
        for path in args.corpus_data or [args.data]:
            corpus_examples.extend(load_examples(path))
        docs = build_corpus(corpus_examples, args.context_dir, args.max_context_chars)
        if args.retriever == "bm25":
            retriever = BM25Retriever(docs)
        else:
            retriever = SentenceBertRetriever(docs, args.sbert_model, args.batch_size, args.device)

    rows = []
    for i, ex in enumerate(examples, start=1):
        if args.retriever == "gold":
            contexts = gold_contexts(ex, args.context_dir, args.max_context_chars)
            retrieved = [{"text": text, "content": "gold"} for text in contexts[: args.top_k]]
        else:
            retrieved = retriever.search(ex.get("question", ""), args.top_k) if retriever else []
        context = "\n\n".join(doc["text"] for doc in retrieved)
        prediction = extract_answer(
            ex.get("question", ""),
            context,
            tokenizer,
            model,
            torch,
            device,
            args.max_length,
            args.max_answer_length,
        )
        rows.append(
            {
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "reference": ex.get("answer", ""),
                "prediction": prediction,
                "model": f"EQUALS:{args.retriever}+MRC",
                "retrieved": [doc.get("content") for doc in retrieved],
            }
        )
        if i % 50 == 0 or i == len(examples):
            print(f"Processed {i}/{len(examples)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
