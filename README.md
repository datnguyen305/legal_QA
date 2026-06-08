# Legal QA Model Runs

This workspace contains Vietnamese legal QA splits in `dataset/`, legal context
files in `dataset/contexts/`, and paper PDFs in `papers/`. The scripts below run
the implemented paper models and evaluate generated answers with ROUGE-L,
METEOR-style exact token matching, and CIDEr.

## Setup

Install model dependencies first:

```bash
python3 -m pip install -r requirements-models.txt
```

## Implemented Paper Models

Only the proposed model paths are implemented. Competitive models from the
papers are not implemented.

## Fair Comparison Groups

Use `configs/comparison_groups.json` as the comparison taxonomy. Do not rank all
models in one flat table without noting the group, because the systems differ in
pretraining, answer style, and retrieval assumptions.

Recommended direct-comparison groups:

- Pretrained extractive readers: `EQUALS`, `FETSF-MRC`, `SAE`, `RE3`.
- Custom extractive readers trained from scratch: `TD-SAN`, `Deep Cascade`.
- Custom generative reader trained from scratch: `CPG`.
- Oracle/sanity checks: `evaluation --upper-bound`, `baseline paper_gold`.

For fair reported scores, keep these controls the same within a table:

- same train/dev/test split and same `dataset/contexts`;
- same `TRAIN_LIMIT`, `DEV_LIMIT`, and `LIMIT` if using smoke runs;
- same evaluation mode, normally prediction mode (`EVAL_UPPER_BOUND=0`);
- same base model for pretrained systems, normally `bert-base-multilingual-cased`;
- retriever setting reported explicitly, especially for EQUALS `gold`, `bm25`,
  or `sbert`.

## Repository Layout

- `configs/models/`: JSON config defaults for each implemented paper model.
- `configs/comparison_groups.json`: fair-comparison group definitions.
- `scripts/model_architectures/`: proposed paper model architectures.
- `scripts/data_preprocessing/`: dataset loading, context handling, and model-specific preprocessing.
- `scripts/pipelines/`: one-command bash pipelines for training, prediction, and evaluation.
- `scripts/train_*.py` and `scripts/run_*.py`: model training and inference entrypoints.
- `scripts/evaluate_predictions.py`: ROUGE-L, METEOR-style, and CIDEr evaluation.

The bash pipelines currently read settings from environment variables. The JSON
files in `configs/models/` mirror those defaults so model settings can be
reviewed and versioned in one place.

## One-Command Runs

Each pipeline trains, predicts, and evaluates. By default, pipeline evaluation
scores the model's actual predictions (`EVAL_UPPER_BOUND=0`).

```bash
bash scripts/pipelines/run_equals_pipeline.sh
bash scripts/pipelines/run_fetsf_mrc_pipeline.sh
bash scripts/pipelines/run_sae_pipeline.sh
bash scripts/pipelines/run_tdsan_pipeline.sh
bash scripts/pipelines/run_deep_cascade_pipeline.sh
bash scripts/pipelines/run_cpg_pipeline.sh
bash scripts/pipelines/run_re3_pipeline.sh
```

Useful smoke-run overrides:

```bash
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_equals_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_fetsf_mrc_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_sae_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_tdsan_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_deep_cascade_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_cpg_pipeline.sh
LIMIT=100 TRAIN_LIMIT=1000 DEV_LIMIT=200 bash scripts/pipelines/run_re3_pipeline.sh
```

Useful inference-only overrides after training:

```bash
SKIP_TRAIN=1 RETRIEVER=bm25 DEVICE=cuda bash scripts/pipelines/run_equals_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_fetsf_mrc_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_sae_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_tdsan_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_deep_cascade_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_cpg_pipeline.sh
SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_re3_pipeline.sh
```

To evaluate the upper-bound oracle instead of actual model predictions, set
`EVAL_UPPER_BOUND=1`:

```bash
EVAL_UPPER_BOUND=1 SKIP_TRAIN=1 DEVICE=cuda bash scripts/pipelines/run_equals_pipeline.sh
```

### EQUALS

- `BM25 + MRC`
- `Sentence-BERT + MRC`
- `Gold + MRC` using the annotated article as context

Train the MRC component:

```bash
python3 scripts/train_equals_mrc.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --base-model bert-base-multilingual-cased \
  --output-dir models/equals_mrc \
  --batch-size 4 \
  --epochs 2
```

Run with gold contexts:

```bash
python3 scripts/run_equals.py \
  --retriever gold \
  --qa-model models/equals_mrc \
  --data dataset/test_data.json \
  --output outputs/equals_gold_mrc.jsonl
```

Run the full BM25 retrieval + MRC pipeline:

```bash
python3 scripts/run_equals.py \
  --retriever bm25 \
  --qa-model models/equals_mrc \
  --corpus-data dataset/train_data.json dataset/dev_data.json dataset/test_data.json \
  --data dataset/test_data.json \
  --top-k 1 \
  --output outputs/equals_bm25_mrc.jsonl
```

Run the full Sentence-BERT retrieval + MRC pipeline:

```bash
python3 scripts/run_equals.py \
  --retriever sbert \
  --qa-model models/equals_mrc \
  --sbert-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --corpus-data dataset/train_data.json dataset/dev_data.json dataset/test_data.json \
  --data dataset/test_data.json \
  --top-k 1 \
  --output outputs/equals_sbert_mrc.jsonl
```

The EQUALS bash pipeline defaults to `RETRIEVER=gold`. Set `RETRIEVER=bm25` or
`RETRIEVER=sbert` to run the full retrieval + MRC variants.

### FETSF-MRC

The FETSF-MRC proposed model is implemented with a scanning
module, evidence sentence prediction, evidence-weighted detailed reading, span
prediction, and feedback loss.

Train:

```bash
python3 scripts/train_fetsf_mrc.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --base-model bert-base-multilingual-cased \
  --output-dir models/fetsf_mrc \
  --batch-size 4 \
  --epochs 2
```

Run:

```bash
python3 scripts/run_fetsf_mrc.py \
  --model-dir models/fetsf_mrc \
  --data dataset/test_data.json \
  --output outputs/fetsf_mrc.jsonl
```

### Select, Answer and Explain

The AAAI-20 Select, Answer and Explain proposed system is implemented with:

- document selection using BERT CLS embeddings, multi-head self-attention, and pairwise ranking loss;
- answer span prediction;
- mixed attentive sentence pooling from span logits and token self-attention;
- GNN-based supporting sentence prediction;
- answer type classification.

Train:

```bash
python3 scripts/train_sae.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --base-model bert-base-multilingual-cased \
  --output-dir models/sae
```

Run:

```bash
python3 scripts/run_sae.py \
  --model-dir models/sae \
  --data dataset/test_data.json \
  --corpus-data dataset/train_data.json dataset/dev_data.json dataset/test_data.json \
  --output outputs/sae.jsonl
```

### Token-Level Dynamic Self-Attention Network

The TD-SAN / DynSAN paper is implemented with:

- local depthwise separable convolution encoders;
- gated top-K token selection;
- dynamic self-attention over selected tokens;
- question-to-passage alignment;
- token-level cross-passage dynamic self-attention;
- extractive start/end span prediction with gate sparsity regularization.

Train:

```bash
python3 scripts/train_tdsan.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --output-dir models/tdsan
```

Run:

```bash
python3 scripts/run_tdsan.py \
  --model-dir models/tdsan \
  --data dataset/test_data.json \
  --output outputs/tdsan.jsonl
```

### Deep Cascade Model

The AAAI-19 Deep Cascade Model is implemented with:

- lightweight document ranking over textual overlap and structure features;
- lightweight paragraph ranking over question/paragraph and paragraph-position features;
- a shared deep attention reader with question/document co-attention;
- auxiliary document extraction and paragraph extraction losses;
- final answer span extraction over selected document content.

Train:

```bash
python3 scripts/train_deep_cascade.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --output-dir models/deep_cascade
```

Run:

```bash
python3 scripts/run_deep_cascade.py \
  --model-dir models/deep_cascade \
  --data dataset/test_data.json \
  --output outputs/deep_cascade.jsonl
```

### Curriculum Pointer-Generator Networks

The long-narrative IAL-CPG paper is implemented as a generative reader with:

- curriculum retrieval over context chunks using answer-based easy views and question-based hard views;
- dynamic chunk sizes for understandability;
- an Introspective Alignment Layer with local block self-attention over decomposed context-question alignments;
- an LSTM pointer-generator decoder that can either copy from context or generate from vocabulary.

Train:

```bash
python3 scripts/train_cpg.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --output-dir models/cpg
```

Run:

```bash
python3 scripts/run_cpg.py \
  --model-dir models/cpg \
  --data dataset/test_data.json \
  --output outputs/cpg.jsonl
```

### Retrieve, Read, Rerank

The RE3QA paper is implemented with:

- segment construction with sliding windows;
- early-stopped neural retrieval from an intermediate Transformer hidden layer;
- distantly supervised span reader from final hidden states;
- span-level non-overlap candidate pruning;
- span representation reranker;
- final weighted retrieval + reading + reranking score.

Train:

```bash
python3 scripts/train_re3.py \
  --train-data dataset/train_data.json \
  --dev-data dataset/dev_data.json \
  --base-model bert-base-multilingual-cased \
  --output-dir models/re3
```

Run:

```bash
python3 scripts/run_re3.py \
  --model-dir models/re3 \
  --data dataset/test_data.json \
  --output outputs/re3.jsonl
```

## Evaluation

Evaluate any prediction JSONL produced by a model:

```bash
python3 scripts/evaluate_predictions.py \
  --predictions outputs/equals_gold_mrc.jsonl \
  --output outputs/equals_gold_mrc_metrics.json
```

Evaluate the upper-bound score for the same file by replacing each prediction
with its reference answer during scoring:

```bash
python3 scripts/evaluate_predictions.py \
  --predictions outputs/equals_gold_mrc.jsonl \
  --upper-bound \
  --output outputs/equals_gold_mrc_upper_bound_metrics.json
```

## Context Handling

The legal context files are expected under `dataset/contexts/`. Each split item
references a file through `contexts.*.content`; the runners load that file's
`passage` field.

For extractive MRC training, the loaders prefer article-aware slicing when the
split metadata or answer text cites `Điều ...`. This keeps the relevant article
near the front of the model input. Examples are used for extractive training only
when a reference answer span can be found in the loaded context.

## Baseline Smoke Test

This is only a sanity check for data loading and metrics, not a paper model.

```bash
python3 scripts/run_baseline.py \
  --data dataset/test_data.json \
  --baseline context \
  --limit 100 \
  --output outputs/context_100.jsonl

python3 scripts/evaluate_predictions.py \
  --predictions outputs/context_100.jsonl \
  --output outputs/context_100_metrics.json
```

Use `--baseline paper_gold` as an oracle sanity check. It should score near the
maximum because it copies the reference answer.

## Run A Seq2Seq Model

Install model dependencies first:

```bash
python3 -m pip install -r requirements-models.txt
```

Then run a local model path or Hugging Face model id:

```bash
python3 scripts/run_hf_seq2seq.py \
  --model VietAI/vit5-base \
  --data dataset/test_data.json \
  --limit 100 \
  --batch-size 4 \
  --max-context-chars 12000 \
  --output outputs/vit5_100.jsonl

python3 scripts/evaluate_predictions.py \
  --predictions outputs/vit5_100.jsonl \
  --output outputs/vit5_100_metrics.json
```

For a full test split, remove `--limit`. Increase `--max-length` and
`--max-context-chars` only if the selected model supports longer inputs.
