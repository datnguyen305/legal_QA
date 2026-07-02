#!/usr/bin/env python3
"""BM25 retrieval over legal context files with set-F1 evaluation."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def load_split(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    return rows[:limit] if limit is not None else rows


def context_refs(example: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    contexts = example.get("contexts") or {}
    values = contexts.values() if isinstance(contexts, dict) else contexts
    for ctx in values:
        if isinstance(ctx, dict) and ctx.get("content"):
            refs.add(str(ctx["content"]))
    return refs


def url_slug(link: str) -> str:
    slug = urllib.parse.urlparse(link or "").path.rsplit("/", 1)[-1]
    return slug[:-5] if slug.endswith(".aspx") else slug


def context_key(data: dict[str, Any]) -> str:
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return url_slug(str(data.get("link", "")))


def fast_context_key(path: Path) -> str:
    fallback = ""
    with path.open(encoding="utf-8") as f:
        for _, line in zip(range(5), f):
            stripped = line.strip()
            if stripped.startswith('"name"'):
                return json.loads(stripped.split(":", 1)[1].rstrip(",").strip()).strip()
            if stripped.startswith('"link"'):
                fallback = url_slug(json.loads(stripped.split(":", 1)[1].rstrip(",").strip()))
    return fallback


def collect_referenced_contexts(split_paths: list[Path], limit: int | None) -> set[str]:
    refs: set[str] = set()
    for split_path in split_paths:
        for row in load_split(split_path, limit):
            refs.update(context_refs(row))
    return refs


def structured_context_files(context_dir: Path, structured_dir: Path) -> list[Path]:
    structured_keys = {path.stem for path in structured_dir.glob("*/*.json")}
    paths: list[Path] = []
    for path in context_dir.glob("context_*.json"):
        if fast_context_key(path) in structured_keys:
            paths.append(path)
    return sorted(paths)


def load_corpus(
    context_dir: Path,
    structured_dir: Path,
    split_paths: list[Path],
    scope: str,
    limit: int | None,
    corpus_limit: int | None,
) -> list[dict[str, str]]:
    if scope == "referenced":
        paths = sorted(context_dir / ref for ref in collect_referenced_contexts(split_paths, limit))
    elif scope == "contexts":
        paths = sorted(context_dir.glob("context_*.json"))
    else:
        paths = structured_context_files(context_dir, structured_dir)
    if corpus_limit is not None:
        paths = paths[:corpus_limit]

    docs: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        passage = data.get("passage")
        if not isinstance(passage, str) or not passage.strip():
            continue
        docs.append(
            {
                "content": path.name,
                "name": str(data.get("name") or context_key(data)),
                "link": str(data.get("link") or ""),
                "text": passage,
            }
        )
    return docs


class BM25Index:
    def __init__(self, docs: list[dict[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        dfs: Counter[str] = Counter()
        for doc_id, doc in enumerate(self.docs):
            tokens = tokenize(doc["name"] + " " + doc["text"])
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                self.postings[term].append((doc_id, tf))
                dfs[term] += 1
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        doc_count = len(self.docs)
        self.idf = {
            term: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def score_dict(self, query: str) -> dict[int, float]:
        query_terms = Counter(tokenize(query))
        scores: defaultdict[int, float] = defaultdict(float)
        for term, qtf in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings.get(term, []):
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[doc_id] += qtf * idf * (tf * (self.k1 + 1.0)) / denom
        return dict(scores)

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        scores = self.score_dict(query)
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [(score, self.docs[doc_id]) for doc_id, score in best]


class DenseLSAIndex:
    def __init__(
        self,
        docs: list[dict[str, str]],
        n_components: int = 256,
        max_features: int = 100000,
        max_doc_chars: int = 12000,
    ) -> None:
        self.docs = docs
        self.max_doc_chars = max_doc_chars
        texts = [self._doc_text(doc) for doc in docs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=max_features,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        components = min(n_components, max(1, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=13)
        self.doc_embeddings = normalize(self.svd.fit_transform(tfidf), norm="l2").astype(np.float32)

    def _doc_text(self, doc: dict[str, str]) -> str:
        return f"{doc['name']} {doc['text'][: self.max_doc_chars]}"

    def score_dict(self, query: str, candidate_count: int) -> dict[int, float]:
        if not query.strip():
            return {}
        query_tfidf = self.vectorizer.transform([query])
        query_embedding = normalize(self.svd.transform(query_tfidf), norm="l2").astype(np.float32)[0]
        scores = self.doc_embeddings @ query_embedding
        top_n = min(candidate_count, len(scores))
        if top_n <= 0:
            return {}
        if top_n == len(scores):
            candidate_ids = np.arange(len(scores))
        else:
            candidate_ids = np.argpartition(scores, -top_n)[-top_n:]
        return {int(doc_id): float(scores[doc_id]) for doc_id in candidate_ids}

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        scores = self.score_dict(query, top_k)
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [(score, self.docs[doc_id]) for doc_id, score in best]


def normalize_candidate_scores(scores: dict[int, float], candidates: set[int]) -> dict[int, float]:
    values = [scores.get(doc_id, 0.0) for doc_id in candidates]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if high <= low:
        return {doc_id: 0.0 for doc_id in candidates}
    return {doc_id: (scores.get(doc_id, 0.0) - low) / (high - low) for doc_id in candidates}


class HybridBM25DenseIndex:
    def __init__(
        self,
        docs: list[dict[str, str]],
        bm25: BM25Index,
        dense: DenseLSAIndex,
        bm25_weight: float = 0.5,
        candidate_count: int = 100,
    ) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = bm25_weight
        self.candidate_count = candidate_count

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        bm25_scores = self.bm25.score_dict(query)
        bm25_top = dict(heapq.nlargest(self.candidate_count, bm25_scores.items(), key=lambda item: item[1]))
        dense_top = self.dense.score_dict(query, self.candidate_count)
        candidates = set(bm25_top) | set(dense_top)
        bm25_norm = normalize_candidate_scores(bm25_top, candidates)
        dense_norm = normalize_candidate_scores(dense_top, candidates)
        dense_weight = 1.0 - self.bm25_weight
        scores = {
            doc_id: self.bm25_weight * bm25_norm.get(doc_id, 0.0)
            + dense_weight * dense_norm.get(doc_id, 0.0)
            for doc_id in candidates
        }
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [(score, self.docs[doc_id]) for doc_id, score in best]


def evaluate_split(
    rows: list[dict[str, Any]],
    index: Any,
    top_k: int,
    output_path: Path | None,
    progress_label: str | None = None,
) -> dict[str, float]:
    total = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    hit_at_1 = 0
    mrr_sum = 0.0

    writer = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for idx, row in enumerate(rows):
            if progress_label and (idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == len(rows)):
                done = idx + 1
                width = 30
                filled = int(width * done / max(1, len(rows)))
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r{progress_label}: [{bar}] {done}/{len(rows)}", end="", file=sys.stderr, flush=True)
            gold = context_refs(row)
            if not gold:
                continue
            retrieved = index.search(str(row.get("question", "")), top_k)
            retrieved_ids = [doc["content"] for _, doc in retrieved]
            hits = len(set(retrieved_ids) & gold)
            precision = hits / max(1, len(retrieved_ids))
            recall = hits / len(gold)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            first_hit_rank = next((rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in gold), None)

            total += 1
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            hit_at_1 += int(bool(retrieved_ids and retrieved_ids[0] in gold))
            mrr_sum += 0.0 if first_hit_rank is None else 1.0 / first_hit_rank

            if writer:
                writer.write(
                    json.dumps(
                        {
                            "id": row.get("id", str(idx)),
                            "question": row.get("question", ""),
                            "gold_contexts": sorted(gold),
                            "retrieved_contexts": [
                                {
                                    "rank": rank,
                                    "score": score,
                                    "content": doc["content"],
                                    "name": doc["name"],
                                    "link": doc["link"],
                                }
                                for rank, (score, doc) in enumerate(retrieved, 1)
                            ],
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "first_hit_rank": first_hit_rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if progress_label:
            print(file=sys.stderr)
        if writer:
            writer.close()

    return {
        "examples": float(total),
        f"precision@{top_k}": precision_sum / max(1, total),
        f"recall@{top_k}": recall_sum / max(1, total),
        "f1": f1_sum / max(1, total),
        "hit@1": hit_at_1 / max(1, total),
        "mrr": mrr_sum / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/structured-single-hop-IR")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--structured-dir", default="dataset/structured-single-hop-IR/structured_data")
    parser.add_argument("--splits", nargs="+", default=["train_data.json", "dev_data.json", "test_data.json"])
    parser.add_argument("--corpus-scope", choices=["contexts", "structured", "referenced"], default="structured")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corpus-limit", type=int, default=None)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--dense-components", type=int, default=256)
    parser.add_argument("--dense-max-features", type=int, default=100000)
    parser.add_argument("--dense-max-doc-chars", type=int, default=12000)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-candidates", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/bm25")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    context_dir = Path(args.context_dir)
    structured_dir = Path(args.structured_dir)
    split_paths = [data_dir / split for split in args.splits]

    docs = load_corpus(context_dir, structured_dir, split_paths, args.corpus_scope, args.limit, args.corpus_limit)
    bm25_index = BM25Index(docs, k1=args.k1, b=args.b)
    if args.retriever == "bm25":
        index = bm25_index
    else:
        dense_index = DenseLSAIndex(
            docs,
            n_components=args.dense_components,
            max_features=args.dense_max_features,
            max_doc_chars=args.dense_max_doc_chars,
        )
        if args.retriever == "dense":
            index = dense_index
        else:
            index = HybridBM25DenseIndex(
                docs,
                bm25_index,
                dense_index,
                bm25_weight=args.hybrid_bm25_weight,
                candidate_count=args.hybrid_candidates,
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "corpus_scope": args.corpus_scope,
        "retriever": args.retriever,
        "documents": len(docs),
        "top_k": args.top_k,
        "k1": args.k1,
        "b": args.b,
        "dense_components": args.dense_components if args.retriever in {"dense", "hybrid"} else None,
        "hybrid_bm25_weight": args.hybrid_bm25_weight if args.retriever == "hybrid" else None,
        "hybrid_candidates": args.hybrid_candidates if args.retriever == "hybrid" else None,
        "splits": {},
    }

    for split_path in split_paths:
        rows = load_split(split_path, args.limit)
        pred_path = None if args.no_predictions else output_dir / f"{split_path.stem}_{args.retriever}_top{args.top_k}.jsonl"
        label = None if args.quiet else f"{args.retriever} {split_path.name}"
        summary["splits"][split_path.name] = evaluate_split(rows, index, args.top_k, pred_path, label)

    summary_path = output_dir / f"{args.retriever}_{args.corpus_scope}_top{args.top_k}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
