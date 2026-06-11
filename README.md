# Legal QA Proposed Methods

This workspace contains runnable implementations for three proposed QA methods:

- **CPG**: curriculum pointer-generator with an Introspective Alignment Layer.
- **S-NET**: evidence extraction followed by GRU seq2seq answer synthesis with evidence start/end feature embeddings, trained from scratch.
- **LatentQA**: stochastic selector network that marginalizes answer tokens over vocabulary, question-copy, and context-copy sources.

The processed dataset files are expected at:

- `dataset/train_data.json`
- `dataset/dev_data.json`
- `dataset/test_data.json`
- `dataset/contexts/`

Each prediction file is JSONL with `prediction` and `reference` fields. The evaluator reports:

- `bleu_4`
- `rouge_l`
- `meteor`
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1` when `--bertscore` is enabled

## Install

```bash
python3 -m pip install -r requirements-models.txt
```

## Smoke Test

Use small limits before full training:

```bash
TRAIN_LIMIT=100 DEV_LIMIT=20 TEST_LIMIT=20 BERTSCORE=0 \
  scripts/pipelines/run_cpg_snet_latentqa.sh
```

## Full Run

```bash
BERTSCORE=1 scripts/pipelines/run_cpg_snet_latentqa.sh
```

## H100 60GB Full Run

The default pipeline settings are tuned for a single H100 60GB:

```bash
DEVICE=cuda AMP=bf16 NUM_WORKERS=16 BERTSCORE=0 \
  scripts/pipelines/run_cpg_snet_latentqa.sh
```

The concrete values are recorded in `configs/h100_full_training.json`.

Metrics are written to:

- `outputs/cpg_metrics.json`
- `outputs/snet_metrics.json`
- `outputs/latentqa_metrics.json`

Predictions are written to:

- `outputs/cpg_predictions.jsonl`
- `outputs/snet_predictions.jsonl`
- `outputs/latentqa_predictions.jsonl`
