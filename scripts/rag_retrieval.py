#!/usr/bin/env python3
"""Local RAG-style retrieval systems for the legal IR dataset.

The implementations are practical, dependency-light variants of the systems in
``papers/IR``. They share the same corpus loading and evaluation format as
``bm25_retrieval.py`` and are intended for controlled retrieval experiments on
this repository's structured legal context corpus.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bm25_retrieval import (  # noqa: E402
    BM25Index,
    DenseLSAIndex,
    HybridBM25DenseIndex,
    context_refs,
    load_corpus,
    load_split,
    normalize_candidate_scores,
    tokenize,
)


def top_items(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    return heapq.nlargest(k, scores.items(), key=lambda item: item[1])


def normalized_fusion(score_maps: list[tuple[dict[int, float], float]]) -> dict[int, float]:
    candidates: set[int] = set()
    for scores, _ in score_maps:
        candidates.update(scores)
    fused = {doc_id: 0.0 for doc_id in candidates}
    for scores, weight in score_maps:
        normed = normalize_candidate_scores(scores, candidates)
        for doc_id, score in normed.items():
            fused[doc_id] += weight * score
    return fused


class IRCoTRetriever:
    """Iterative retrieval with query expansion from retrieved rationales.

    This approximates IRCoT without requiring an LLM: each iteration retrieves
    evidence, extracts high-signal terms from it, appends them to the query, and
    reruns BM25. Final scores are the accumulated evidence scores.
    """

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, iterations: int = 2, expansion_terms: int = 8) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.iterations = iterations
        self.expansion_terms = expansion_terms

    def _expand_terms(self, doc_ids: list[int], query_terms: set[str]) -> list[str]:
        counts: Counter[str] = Counter()
        for doc_id in doc_ids:
            counts.update(t for t in tokenize(self.docs[doc_id]["text"][:6000]) if len(t) > 2 and t not in query_terms)
        weighted = {
            term: count * self.bm25.idf.get(term, 0.0)
            for term, count in counts.items()
            if term in self.bm25.idf
        }
        return [term for term, _ in top_items(weighted, self.expansion_terms)]

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        expanded = query
        accumulated: defaultdict[int, float] = defaultdict(float)
        for step in range(self.iterations):
            scores = self.bm25.score_dict(expanded)
            for doc_id, score in scores.items():
                accumulated[doc_id] += score / (step + 1)
            seeds = [doc_id for doc_id, _ in top_items(scores, max(top_k, 5))]
            query_terms = set(tokenize(expanded))
            additions = self._expand_terms(seeds, query_terms)
            if not additions:
                break
            expanded = f"{expanded} {' '.join(additions)}"
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(dict(accumulated), top_k)]


class HippoRAGRetriever:
    """Graph-augmented retrieval with personalized propagation.

    Documents are connected to salient terms. Query terms and BM25 seed docs
    activate term/document nodes; scores propagate through the bipartite graph.
    """

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, max_terms_per_doc: int = 40, seed_docs: int = 20) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.seed_docs = seed_docs
        self.doc_terms: list[list[str]] = []
        self.term_docs: defaultdict[str, list[int]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            counts = Counter(t for t in tokenize(f"{doc['name']} {doc['text'][:8000]}") if len(t) > 2)
            weighted = {
                term: count * bm25.idf.get(term, 0.0)
                for term, count in counts.items()
                if term in bm25.idf
            }
            terms = [term for term, _ in top_items(weighted, max_terms_per_doc)]
            self.doc_terms.append(terms)
            for term in terms:
                self.term_docs[term].append(doc_id)

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        bm25_scores = self.bm25.score_dict(query)
        doc_scores: defaultdict[int, float] = defaultdict(float)
        term_scores: defaultdict[str, float] = defaultdict(float)
        for term in tokenize(query):
            if term in self.term_docs:
                term_scores[term] += self.bm25.idf.get(term, 1.0)
        for doc_id, score in top_items(bm25_scores, self.seed_docs):
            doc_scores[doc_id] += score
            for term in self.doc_terms[doc_id]:
                term_scores[term] += 0.15 * score / max(1, len(self.doc_terms[doc_id]))
        for term, score in list(term_scores.items()):
            neighbors = self.term_docs.get(term, [])
            if not neighbors:
                continue
            for doc_id in neighbors[:200]:
                doc_scores[doc_id] += score / math.sqrt(len(neighbors))
        fused = normalized_fusion([(bm25_scores, 0.55), (dict(doc_scores), 0.45)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


class LightRAGRetriever:
    """Lightweight hybrid retrieval with graph-style neighbor expansion."""

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, dense: DenseLSAIndex, candidate_count: int = 80) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.dense = dense
        self.candidate_count = candidate_count

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        bm25_scores = dict(top_items(self.bm25.score_dict(query), self.candidate_count))
        dense_scores = self.dense.score_dict(query, self.candidate_count)
        query_terms = set(tokenize(query))
        neighbor_scores: defaultdict[int, float] = defaultdict(float)
        for doc_id in set(bm25_scores) | set(dense_scores):
            doc_terms = set(tokenize(self.docs[doc_id]["name"]))
            overlap = len(query_terms & doc_terms)
            if overlap:
                neighbor_scores[doc_id] += overlap
        fused = normalized_fusion([(bm25_scores, 0.45), (dense_scores, 0.45), (dict(neighbor_scores), 0.10)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


class MiniRAGRetriever:
    """Small-footprint retrieval over compressed document representations."""

    def __init__(self, docs: list[dict[str, str]], n_components: int = 96) -> None:
        compact_docs = [
            {**doc, "text": doc["text"][:2500]}
            for doc in docs
        ]
        self.docs = docs
        self.bm25 = BM25Index(compact_docs)
        self.dense = DenseLSAIndex(compact_docs, n_components=n_components, max_features=40000, max_doc_chars=2500)
        self.hybrid = HybridBM25DenseIndex(compact_docs, self.bm25, self.dense, bm25_weight=0.65, candidate_count=60)

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        results = self.hybrid.search(query, top_k)
        return [(score, self.docs[int(doc["content"].split("_")[-1].split(".")[0])] if False else doc) for score, doc in results]


class RAPTORRetriever:
    """Hierarchical cluster-first retrieval over dense document embeddings."""

    def __init__(
        self,
        docs: list[dict[str, str]],
        bm25: BM25Index,
        dense: DenseLSAIndex,
        clusters: int = 64,
        top_clusters: int = 4,
    ) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.dense = dense
        n_clusters = min(clusters, max(1, len(docs) // 2))
        self.top_clusters = min(top_clusters, n_clusters)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=13, batch_size=512, n_init="auto")
        self.labels = self.kmeans.fit_predict(self.dense.doc_embeddings)
        self.cluster_docs: defaultdict[int, list[int]] = defaultdict(list)
        for doc_id, label in enumerate(self.labels):
            self.cluster_docs[int(label)].append(doc_id)
        self.cluster_embeddings = normalize(self.kmeans.cluster_centers_, norm="l2").astype(np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        query_tfidf = self.dense.vectorizer.transform([query])
        query_embedding = normalize(self.dense.svd.transform(query_tfidf), norm="l2").astype(np.float32)[0]
        cluster_scores = self.cluster_embeddings @ query_embedding
        cluster_ids = np.argpartition(cluster_scores, -self.top_clusters)[-self.top_clusters:]
        candidates = {doc_id for cluster_id in cluster_ids for doc_id in self.cluster_docs[int(cluster_id)]}
        bm25_scores = self.bm25.score_dict(query)
        dense_scores = {doc_id: float(self.dense.doc_embeddings[doc_id] @ query_embedding) for doc_id in candidates}
        scoped_bm25 = {doc_id: bm25_scores.get(doc_id, 0.0) for doc_id in candidates}
        fused = normalized_fusion([(scoped_bm25, 0.50), (dense_scores, 0.50)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


class ViHERMESRetriever:
    """Vietnamese legal hybrid retrieval baseline using metadata expansion."""

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, dense: DenseLSAIndex) -> None:
        self.docs = docs
        self.hybrid = HybridBM25DenseIndex(docs, bm25, dense, bm25_weight=0.7, candidate_count=120)

    def _normalize_query(self, query: str) -> str:
        expansions = []
        lower = query.lower()
        if "nghị định" in lower:
            expansions.append("nghị định chính phủ điều khoản")
        if "luật" in lower:
            expansions.append("luật quốc hội điều khoản")
        if "thông tư" in lower:
            expansions.append("thông tư bộ điều khoản")
        return f"{query} {' '.join(expansions)}"

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        return self.hybrid.search(self._normalize_query(query), top_k)


def evaluate_split(rows: list[dict[str, Any]], index: Any, top_k: int, output_path: Path | None) -> dict[str, float]:
    total = 0
    precision_sum = recall_sum = f1_sum = 0.0
    hit_at_1 = 0
    mrr_sum = 0.0
    writer = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for idx, row in enumerate(rows):
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
                writer.write(json.dumps({
                    "id": row.get("id", str(idx)),
                    "question": row.get("question", ""),
                    "gold_contexts": sorted(gold),
                    "retrieved_contexts": [
                        {"rank": rank, "score": score, "content": doc["content"], "name": doc["name"], "link": doc["link"]}
                        for rank, (score, doc) in enumerate(retrieved, 1)
                    ],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "first_hit_rank": first_hit_rank,
                }, ensure_ascii=False) + "\n")
    finally:
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


def make_retriever(args: argparse.Namespace, docs: list[dict[str, str]]) -> Any:
    bm25 = BM25Index(docs, k1=args.k1, b=args.b)
    needs_dense = args.method in {"lightrag", "minirag", "raptor", "vi_hermes"}
    dense = None
    if needs_dense:
        dense = DenseLSAIndex(
            docs,
            n_components=args.dense_components,
            max_features=args.dense_max_features,
            max_doc_chars=args.dense_max_doc_chars,
        )
    if args.method == "ircot":
        return IRCoTRetriever(docs, bm25, iterations=args.ircot_iterations, expansion_terms=args.ircot_expansion_terms)
    if args.method == "hipporag":
        return HippoRAGRetriever(docs, bm25, max_terms_per_doc=args.graph_terms_per_doc, seed_docs=args.graph_seed_docs)
    if args.method == "lightrag":
        return LightRAGRetriever(docs, bm25, dense, candidate_count=args.hybrid_candidates)
    if args.method == "minirag":
        return MiniRAGRetriever(docs, n_components=min(args.dense_components, 96))
    if args.method == "raptor":
        return RAPTORRetriever(docs, bm25, dense, clusters=args.raptor_clusters, top_clusters=args.raptor_top_clusters)
    if args.method == "vi_hermes":
        return ViHERMESRetriever(docs, bm25, dense)
    raise ValueError(f"Unknown RAG method: {args.method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ircot", "hipporag", "lightrag", "minirag", "raptor", "vi_hermes"], required=True)
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
    parser.add_argument("--dense-components", type=int, default=128)
    parser.add_argument("--dense-max-features", type=int, default=100000)
    parser.add_argument("--dense-max-doc-chars", type=int, default=12000)
    parser.add_argument("--hybrid-candidates", type=int, default=100)
    parser.add_argument("--ircot-iterations", type=int, default=2)
    parser.add_argument("--ircot-expansion-terms", type=int, default=8)
    parser.add_argument("--graph-terms-per-doc", type=int, default=40)
    parser.add_argument("--graph-seed-docs", type=int, default=20)
    parser.add_argument("--raptor-clusters", type=int, default=64)
    parser.add_argument("--raptor-top-clusters", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/rag")
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_paths = [data_dir / split for split in args.splits]
    docs = load_corpus(
        Path(args.context_dir),
        Path(args.structured_dir),
        split_paths,
        args.corpus_scope,
        args.limit,
        args.corpus_limit,
    )
    retriever = make_retriever(args, docs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "method": args.method,
        "corpus_scope": args.corpus_scope,
        "documents": len(docs),
        "top_k": args.top_k,
        "splits": {},
    }
    for split_path in split_paths:
        rows = load_split(split_path, args.limit)
        pred_path = None if args.no_predictions else output_dir / f"{split_path.stem}_{args.method}_top{args.top_k}.jsonl"
        summary["splits"][split_path.name] = evaluate_split(rows, retriever, args.top_k, pred_path)
    summary_path = output_dir / f"{args.method}_{args.corpus_scope}_top{args.top_k}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
