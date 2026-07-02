# IR Paper Reproduction Status

This repository includes runnable local retrieval implementations for the IR/RAG
papers under `papers/IR`, but these are not exact full paper reproductions yet.
The current environment does not provide the main dependencies required for full
reproduction:

- no `torch`
- no `transformers`
- no `sentence_transformers`
- no `faiss` / `hnswlib`
- no OpenAI SDK or configured LLM API key

The local implementations therefore use `numpy`, `sklearn`, and `networkx`-style
retrieval components where possible. They are useful baselines on
`dataset/structured-single-hop-IR`, but exact reproduction requires the missing
components listed below.

## Current Runnable Implementations

Entry point:

```bash
python3 scripts/rag_retrieval.py --method <method>
```

Pipeline:

```bash
RAG_METHOD=<method> scripts/pipelines/run_rag_retrieval.sh
```

Methods:

```text
ircot
hipporag
lightrag
minirag
raptor
vi_hermes
```

Default corpus:

```text
dataset/structured-single-hop-IR
dataset/contexts
dataset/structured-single-hop-IR/structured_data
```

Metrics:

```text
precision@3
recall@3
f1
hit@1
mrr
```

## Full Reproduction Requirements

### IRCoT

Paper requirement:

- Initial retrieval from the question.
- LLM generates the next chain-of-thought sentence from question plus retrieved
  evidence.
- The generated CoT sentence becomes the next retrieval query.
- Iterate until answer or max reasoning steps.
- Reader LLM answers from accumulated evidence.

Current local implementation:

- BM25 retrieval.
- Query expansion from high-IDF terms in retrieved evidence.
- No LLM-generated CoT.
- No LLM reader.

Missing for full reproduction:

- LLM provider.
- IRCoT prompt templates.
- Reader prompt and answer generation.
- Optional multi-hop datasets comparable to HotpotQA/2Wiki/MuSiQue/IIRC.

### HippoRAG / HippoRAG 2

Paper requirement:

- LLM/OpenIE extracts triples from passages.
- Build open knowledge graph.
- Embedding model links query entities to graph nodes.
- Personalized PageRank over graph.
- HippoRAG 2 integrates passages into graph retrieval and uses an LLM online to
  filter irrelevant triples/passages.

Current local implementation:

- Salient-term graph over documents.
- BM25 seed retrieval.
- Graph propagation through term-document edges.
- No OpenIE triples.
- No LLM relevance filtering.

Missing for full reproduction:

- OpenIE LLM extraction.
- Entity canonicalization and synonym linking.
- PPR graph over entity/relation triples.
- Online LLM filtering.
- Dense retriever equivalent to the paper setting.

### LightRAG

Paper requirement:

- LLM extracts entities and relationships.
- LLM profiling creates key-value descriptions for entities/relations.
- Deduplication/merge graph nodes and edges.
- Dual-level retrieval over low-level entity/relation details and high-level
  themes.
- Incremental update algorithm.

Current local implementation:

- Sparse+dense hybrid retrieval.
- Lightweight metadata overlap as graph-style expansion.
- No LLM entity/relation extraction.
- No graph key-value profiling.
- No incremental update path.

Missing for full reproduction:

- Entity/relation extraction prompts.
- Graph store.
- KV profiling.
- Low-level/high-level retrieval implementation over the extracted graph.
- Incremental index update logic.

### MiniRAG

Paper requirement:

- Small-language-model-oriented heterogeneous graph indexing.
- Entity and chunk nodes in one graph.
- Lightweight entity extraction.
- Query-driven reasoning path discovery.
- Topology-aware retrieval/ranking.

Current local implementation:

- Compact document representations.
- Hybrid sparse+dense retrieval over shortened text.
- No heterogeneous entity/chunk graph.
- No reasoning path discovery.

Missing for full reproduction:

- Entity extraction component.
- Heterogeneous graph index.
- Entity/chunk node embeddings.
- Path discovery and topology-aware ranking.

### RAPTOR

Paper requirement:

- Split corpus into chunks.
- Embed chunks with SBERT-style embeddings.
- Recursive clustering, originally UMAP + GMM/BIC.
- LLM summarizes each cluster.
- Build tree of summaries and leaves.
- Retrieve using tree traversal or collapsed tree retrieval.

Current local implementation:

- Dense LSA embeddings.
- MiniBatchKMeans clusters.
- Cluster-first reranking.
- No recursive tree.
- No LLM summaries.
- No UMAP/GMM/BIC.

Missing for full reproduction:

- Sentence-transformer embeddings.
- UMAP.
- Gaussian Mixture Model clustering with BIC selection.
- LLM summarizer.
- Recursive tree construction.
- Tree traversal and collapsed-tree retrieval.

### Vi-HERMES

Paper requirement:

- Dataset construction pipeline using semantic clustering and graph-inspired
  sampling.
- Multihop QA generation with structured evidence/reasoning annotations.
- Graph-aware regulatory QA system.
- Legal unit graph with relations such as hierarchy, cross-reference,
  amendment/replacement/supplement.
- Multi-agent QA pipeline for intent, retrieval, and verification.

Current local implementation:

- Vietnamese legal metadata query expansion.
- Hybrid sparse+dense retrieval.
- No healthcare-specific multihop dataset generation.
- No explicit legal relation graph.
- No multi-agent answer verification.

Missing for full reproduction:

- Healthcare regulation corpus and annotations.
- Legal relation extraction/linking.
- Multihop QA generation prompts.
- Regulatory graph-aware retrieval.
- Intent and verification agents.

## What Must Be Added For Exact Reproduction

At minimum:

```bash
python3 -m pip install torch transformers sentence-transformers faiss-cpu umap-learn
```

Optional/depending on method:

```bash
python3 -m pip install openai hnswlib spacy pyvi underthesea
```

Required configuration:

```bash
export OPENAI_API_KEY=...
```

or another local/server LLM provider for:

- IRCoT CoT generation and reader
- HippoRAG OpenIE and online filtering
- LightRAG entity/relation extraction and profiling
- RAPTOR summarization
- Vi-HERMES generation/verification agents

Without these components, exact paper reproduction is not possible in this
workspace; only local approximations and ablations can be run.
