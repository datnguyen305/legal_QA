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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
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
    PyseriniBM25Index,
    context_refs,
    corpus_cache_key,
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


LEGAL_TYPE_PATTERNS = {
    "luat": ("luật", "luat"),
    "bo_luat": ("bộ luật", "bo-luat", "bo luat"),
    "nghi_dinh": ("nghị định", "nghi-dinh", "nghi dinh"),
    "thong_tu": ("thông tư", "thong-tu", "thong tu"),
    "quyet_dinh": ("quyết định", "quyet-dinh", "quyet dinh"),
    "nghi_quyet": ("nghị quyết", "nghi-quyet", "nghi quyet"),
    "chi_thi": ("chỉ thị", "chi-thi", "chi thi"),
    "cong_van": ("công văn", "cong-van", "cong van"),
}
LEGAL_REF_RE = re.compile(
    r"(?i)\b(?:luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|chỉ\s+thị|công\s+văn)\s+"
    r"(?:số\s+)?([0-9]+/[0-9]{4}/[A-ZĐ\-]+)"
)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def legal_doc_type(text: str) -> str | None:
    lowered = text.lower()
    for doc_type, patterns in LEGAL_TYPE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return doc_type
    return None


def legal_year(text: str) -> str | None:
    match = YEAR_RE.search(text)
    return match.group(1) if match else None


def header_terms(text: str, max_terms: int = 8) -> list[str]:
    header = " ".join(line.strip() for line in text[:1200].splitlines()[:12])
    counts = Counter(t for t in tokenize(header) if len(t) > 2)
    return [term for term, _ in counts.most_common(max_terms)]


class LegalGraphIndex:
    """Shared legal graph over documents, salient terms, metadata, and citations."""

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, max_terms_per_doc: int = 40) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.doc_id_by_content = {doc["content"]: doc_id for doc_id, doc in enumerate(docs)}
        self.graph = nx.Graph()
        self.doc_attrs: list[list[str]] = []
        self.attr_docs: defaultdict[str, list[int]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            doc_node = self._doc_node(doc_id)
            self.graph.add_node(doc_node, kind="document", content=doc["content"])
            attrs = self._document_attrs(doc, max_terms_per_doc)
            self.doc_attrs.append(attrs)
            for attr in attrs:
                self.attr_docs[attr].append(doc_id)
                self.graph.add_node(attr, kind=attr.split(":", 1)[0])
                self.graph.add_edge(doc_node, attr)

    def _doc_node(self, doc_id: int) -> str:
        return f"doc:{doc_id}"

    def _document_attrs(self, doc: dict[str, str], max_terms_per_doc: int) -> list[str]:
        text = f"{doc['name']} {doc['text'][:8000]}"
        counts = Counter(t for t in tokenize(text) if len(t) > 2)
        weighted = {
            f"term:{term}": count * self.bm25.idf.get(term, 0.0)
            for term, count in counts.items()
            if term in self.bm25.idf
        }
        attrs = [attr for attr, _ in top_items(weighted, max_terms_per_doc)]
        doc_type = legal_doc_type(f"{doc['name']} {doc['text'][:1000]}")
        if doc_type:
            attrs.append(f"type:{doc_type}")
        year = legal_year(f"{doc['name']} {doc['link']} {doc['text'][:1000]}")
        if year:
            attrs.append(f"year:{year}")
        attrs.extend(f"header:{term}" for term in header_terms(doc["text"]))
        refs = {match.group(1).lower() for match in LEGAL_REF_RE.finditer(doc["text"][:12000])}
        attrs.extend(f"ref:{ref}" for ref in sorted(refs)[:20])
        return list(dict.fromkeys(attrs))

    def query_attrs(self, query: str) -> list[str]:
        attrs = [f"term:{term}" for term in tokenize(query) if f"term:{term}" in self.attr_docs]
        doc_type = legal_doc_type(query)
        if doc_type and f"type:{doc_type}" in self.attr_docs:
            attrs.append(f"type:{doc_type}")
        year = legal_year(query)
        if year and f"year:{year}" in self.attr_docs:
            attrs.append(f"year:{year}")
        refs = {match.group(1).lower() for match in LEGAL_REF_RE.finditer(query)}
        attrs.extend(f"ref:{ref}" for ref in refs if f"ref:{ref}" in self.attr_docs)
        return list(dict.fromkeys(attrs))

    def propagate(
        self,
        query: str,
        seed_scores: dict[int, float],
        candidate_ids: set[int] | None = None,
        seed_weight: float = 0.15,
        max_neighbors_per_attr: int = 250,
    ) -> dict[int, float]:
        attr_scores: defaultdict[str, float] = defaultdict(float)
        for attr in self.query_attrs(query):
            if attr.startswith("term:"):
                attr_scores[attr] += self.bm25.idf.get(attr.split(":", 1)[1], 1.0)
            else:
                attr_scores[attr] += 1.0
        for doc_id, score in top_items(seed_scores, min(50, len(seed_scores))):
            attrs = self.doc_attrs[doc_id]
            for attr in attrs:
                attr_scores[attr] += seed_weight * score / max(1, len(attrs))

        doc_scores: defaultdict[int, float] = defaultdict(float)
        for attr, score in attr_scores.items():
            neighbors = self.attr_docs.get(attr, [])
            if not neighbors:
                continue
            for doc_id in neighbors[:max_neighbors_per_attr]:
                if candidate_ids is not None and doc_id not in candidate_ids:
                    continue
                doc_scores[doc_id] += score / math.sqrt(len(neighbors))
        return dict(doc_scores)


class IRCoTRetriever:
    """Iterative retrieval with query expansion from retrieved rationales.

    This approximates IRCoT without requiring an LLM: each iteration retrieves
    evidence, extracts high-signal terms from it, appends them to the query, and
    reruns BM25. Final scores are the accumulated evidence scores.
    """

    def __init__(
        self,
        docs: list[dict[str, str]],
        bm25: Any,
        iterations: int = 2,
        expansion_terms: int = 8,
        candidate_count: int = 100,
    ) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.iterations = iterations
        self.expansion_terms = expansion_terms
        self.candidate_count = candidate_count
        self.doc_id_by_content = {doc["content"]: doc_id for doc_id, doc in enumerate(docs)}

    def _expand_terms(self, doc_ids: list[int], query_terms: set[str]) -> list[str]:
        counts: Counter[str] = Counter()
        for doc_id in doc_ids:
            counts.update(t for t in tokenize(self.docs[doc_id]["text"][:6000]) if len(t) > 2 and t not in query_terms)
        idf = getattr(self.bm25, "idf", None)
        weighted = {
            term: count * (idf.get(term, 1.0) if idf is not None else 1.0)
            for term, count in counts.items()
            if idf is None or term in idf
        }
        return [term for term, _ in top_items(weighted, self.expansion_terms)]

    def _score_dict(self, query: str) -> dict[int, float]:
        if hasattr(self.bm25, "score_dict"):
            return self.bm25.score_dict(query)
        retrieved = self.bm25.search(query, self.candidate_count)
        scores: dict[int, float] = {}
        for score, doc in retrieved:
            doc_id = self.doc_id_by_content.get(doc["content"])
            if doc_id is not None:
                scores[doc_id] = score
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        expanded = query
        accumulated: defaultdict[int, float] = defaultdict(float)
        for step in range(self.iterations):
            scores = self._score_dict(expanded)
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
    """Graph-augmented retrieval with propagation over a legal document graph."""

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, graph: LegalGraphIndex, seed_docs: int = 20) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.graph = graph
        self.seed_docs = seed_docs

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        bm25_scores = self.bm25.score_dict(query)
        seeds = dict(top_items(bm25_scores, self.seed_docs))
        graph_scores = self.graph.propagate(query, seeds, seed_weight=0.20)
        fused = normalized_fusion([(bm25_scores, 0.50), (graph_scores, 0.50)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


class LightRAGRetriever:
    """Lightweight sparse/dense retrieval with legal-graph neighbor expansion."""

    def __init__(
        self,
        docs: list[dict[str, str]],
        bm25: BM25Index,
        dense: DenseLSAIndex,
        graph: LegalGraphIndex,
        candidate_count: int = 80,
    ) -> None:
        self.docs = docs
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.candidate_count = candidate_count

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        bm25_scores = dict(top_items(self.bm25.score_dict(query), self.candidate_count))
        dense_scores = self.dense.score_dict(query, self.candidate_count)
        candidates = set(bm25_scores) | set(dense_scores)
        graph_scores = self.graph.propagate(query, bm25_scores, candidate_ids=candidates, seed_weight=0.10)
        fused = normalized_fusion([(bm25_scores, 0.40), (dense_scores, 0.40), (graph_scores, 0.20)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


class MiniRAGRetriever:
    """Small-footprint retrieval over compressed text and the legal graph."""

    def __init__(self, docs: list[dict[str, str]], graph: LegalGraphIndex, n_components: int = 96) -> None:
        compact_docs = [
            {**doc, "text": doc["text"][:2500]}
            for doc in docs
        ]
        self.docs = docs
        self.bm25 = BM25Index(compact_docs)
        self.dense = DenseLSAIndex(compact_docs, n_components=n_components, max_features=40000, max_doc_chars=2500)
        self.hybrid = HybridBM25DenseIndex(compact_docs, self.bm25, self.dense, bm25_weight=0.65, candidate_count=60)
        self.graph = graph

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, str]]]:
        sparse_scores = dict(top_items(self.bm25.score_dict(query), 60))
        dense_scores = self.dense.score_dict(query, 60)
        candidates = set(sparse_scores) | set(dense_scores)
        graph_scores = self.graph.propagate(query, sparse_scores, candidate_ids=candidates, seed_weight=0.08)
        fused = normalized_fusion([(sparse_scores, 0.45), (dense_scores, 0.35), (graph_scores, 0.20)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


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
    """Vietnamese legal hybrid retrieval baseline using metadata and graph expansion."""

    def __init__(self, docs: list[dict[str, str]], bm25: BM25Index, dense: DenseLSAIndex, graph: LegalGraphIndex) -> None:
        self.docs = docs
        self.doc_id_by_content = {doc["content"]: doc_id for doc_id, doc in enumerate(docs)}
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
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
        expanded = self._normalize_query(query)
        hybrid_results = self.hybrid.search(expanded, 120)
        hybrid_scores = {
            self.doc_id_by_content[doc["content"]]: score
            for score, doc in hybrid_results
            if doc["content"] in self.doc_id_by_content
        }
        graph_scores = self.graph.propagate(expanded, hybrid_scores, candidate_ids=set(hybrid_scores), seed_weight=0.12)
        fused = normalized_fusion([(hybrid_scores, 0.75), (graph_scores, 0.25)])
        return [(score, self.docs[doc_id]) for doc_id, score in top_items(fused, top_k)]


def evaluate_split(
    rows: list[dict[str, Any]],
    index: Any,
    top_k: int,
    output_path: Path | None,
    progress_label: str | None = None,
) -> dict[str, float]:
    total = 0
    precision_sum = recall_sum = f1_sum = 0.0
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


def make_retriever(args: argparse.Namespace, docs: list[dict[str, str]], output_dir: Path) -> Any:
    method = "ircot" if args.method == "ifcot" else args.method
    if method == "ircot" and args.bm25_backend == "pyserini":
        cache_key = corpus_cache_key(docs, args.corpus_scope)
        bm25 = PyseriniBM25Index(
            docs,
            index_dir=output_dir / "pyserini_indexes" / f"ircot_{cache_key}",
            collection_dir=output_dir / "pyserini_collections" / f"ircot_{cache_key}",
            k1=args.k1,
            b=args.b,
            threads=args.pyserini_threads,
        )
    else:
        bm25 = BM25Index(docs, k1=args.k1, b=args.b)
    graph_methods = {"hipporag", "lightrag", "minirag", "vi_hermes"}
    graph = LegalGraphIndex(docs, bm25, max_terms_per_doc=args.graph_terms_per_doc) if method in graph_methods else None
    needs_dense = method in {"lightrag", "minirag", "raptor", "vi_hermes"}
    dense = None
    if needs_dense:
        dense = DenseLSAIndex(
            docs,
            n_components=args.dense_components,
            max_features=args.dense_max_features,
            max_doc_chars=args.dense_max_doc_chars,
        )
    if method == "ircot":
        return IRCoTRetriever(
            docs,
            bm25,
            iterations=args.ircot_iterations,
            expansion_terms=args.ircot_expansion_terms,
            candidate_count=args.ircot_candidates,
        )
    if method == "hipporag":
        return HippoRAGRetriever(docs, bm25, graph, seed_docs=args.graph_seed_docs)
    if method == "lightrag":
        return LightRAGRetriever(docs, bm25, dense, graph, candidate_count=args.hybrid_candidates)
    if method == "minirag":
        return MiniRAGRetriever(docs, graph, n_components=min(args.dense_components, 96))
    if method == "raptor":
        return RAPTORRetriever(docs, bm25, dense, clusters=args.raptor_clusters, top_clusters=args.raptor_top_clusters)
    if method == "vi_hermes":
        return ViHERMESRetriever(docs, bm25, dense, graph)
    raise ValueError(f"Unknown RAG method: {args.method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ircot", "ifcot", "hipporag", "lightrag", "minirag", "raptor", "vi_hermes"], required=True)
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
    parser.add_argument("--bm25-backend", choices=["python", "pyserini"], default="python")
    parser.add_argument("--pyserini-threads", type=int, default=4)
    parser.add_argument("--dense-components", type=int, default=128)
    parser.add_argument("--dense-max-features", type=int, default=100000)
    parser.add_argument("--dense-max-doc-chars", type=int, default=12000)
    parser.add_argument("--hybrid-candidates", type=int, default=100)
    parser.add_argument("--ircot-iterations", type=int, default=2)
    parser.add_argument("--ircot-expansion-terms", type=int, default=8)
    parser.add_argument("--ircot-candidates", type=int, default=100)
    parser.add_argument("--graph-terms-per-doc", type=int, default=40)
    parser.add_argument("--graph-seed-docs", type=int, default=20)
    parser.add_argument("--raptor-clusters", type=int, default=64)
    parser.add_argument("--raptor-top-clusters", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/rag")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever = make_retriever(args, docs, output_dir)
    summary: dict[str, Any] = {
        "method": args.method,
        "bm25_backend": args.bm25_backend if args.method in {"ircot", "ifcot"} else None,
        "corpus_scope": args.corpus_scope,
        "documents": len(docs),
        "top_k": args.top_k,
        "splits": {},
    }
    legal_graph = getattr(retriever, "graph", None)
    if isinstance(legal_graph, LegalGraphIndex):
        summary["graph"] = {
            "nodes": legal_graph.graph.number_of_nodes(),
            "edges": legal_graph.graph.number_of_edges(),
            "attributes": len(legal_graph.attr_docs),
            "max_terms_per_doc": args.graph_terms_per_doc,
        }
    for split_path in split_paths:
        rows = load_split(split_path, args.limit)
        pred_path = None if args.no_predictions else output_dir / f"{split_path.stem}_{args.method}_top{args.top_k}.jsonl"
        label = None if args.quiet else f"{args.method} {split_path.name}"
        summary["splits"][split_path.name] = evaluate_split(rows, retriever, args.top_k, pred_path, label)
    summary_path = output_dir / f"{args.method}_{args.corpus_scope}_top{args.top_k}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
