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

- `rouge_l`
- `meteor`
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1` by default

Use `BERTSCORE=0` in pipeline scripts, or `--no-bertscore` with `scripts/evaluate_predictions.py`, to disable BERTScore.

Training uses the same stopping policy for all models: save the best checkpoint
by dev ROUGE-L and stop after `patience` epochs without dev ROUGE-L improvement.

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

Run one model at a time:

```bash
TRAIN_LIMIT=100 DEV_LIMIT=20 TEST_LIMIT=20 BERTSCORE=0 \
  scripts/pipelines/run_cpg.sh
```

```bash
TRAIN_LIMIT=100 DEV_LIMIT=20 TEST_LIMIT=20 BERTSCORE=0 \
  scripts/pipelines/run_snet.sh
```

```bash
TRAIN_LIMIT=100 DEV_LIMIT=20 TEST_LIMIT=20 BERTSCORE=0 \
  scripts/pipelines/run_latentqa.sh
```

## Full Run

```bash
BERTSCORE=1 scripts/pipelines/run_cpg_snet_latentqa.sh
```

Individual full runs:

```bash
BERTSCORE=1 scripts/pipelines/run_cpg.sh
```

```bash
BERTSCORE=1 scripts/pipelines/run_snet.sh
```

```bash
BERTSCORE=1 scripts/pipelines/run_latentqa.sh
```

## H100 60GB Full Run

The default pipeline settings are tuned for a single H100 60GB:

```bash
DEVICE=cuda AMP=bf16 NUM_WORKERS=4 \
  scripts/pipelines/run_cpg_snet_latentqa.sh
```

Individual H100 runs:

```bash
DEVICE=cuda AMP=bf16 NUM_WORKERS=4 CPG_EPOCHS=20 CPG_PATIENCE=3 \
  scripts/pipelines/run_cpg.sh
```

```bash
DEVICE=cuda AMP=bf16 NUM_WORKERS=4 SNET_EPOCHS=20 SNET_PATIENCE=3 \
  scripts/pipelines/run_snet.sh
```

```bash
DEVICE=cuda AMP=bf16 NUM_WORKERS=4 LATENTQA_EPOCHS=20 LATENTQA_PATIENCE=3 \
  scripts/pipelines/run_latentqa.sh
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
