# Legal QA Proposed Methods

This workspace contains runnable implementations for three proposed QA methods:

- **CPG**: curriculum pointer-generator with an Introspective Alignment Layer.
- **S-NET**: evidence extraction followed by GRU seq2seq answer synthesis with evidence start/end feature embeddings, trained from scratch.
- **LatentQA**: stochastic selector network that marginalizes answer tokens over vocabulary, question-copy, and context-copy sources.

It also includes pretrained seq2seq fine-tuning pipelines for **ViT5**,
**BARTpho**, **mT5**, and **mBART**.

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

Smoke test a pretrained seq2seq model:

```bash
MODEL_KEY=vit5 TRAIN_LIMIT=100 DEV_LIMIT=20 TEST_LIMIT=20 BERTSCORE=0 \
  scripts/pipelines/run_pretrained_seq2seq.sh
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

Pretrained seq2seq full runs:

```bash
MODEL_KEY=vit5 scripts/pipelines/run_pretrained_seq2seq.sh
```

```bash
MODEL_KEY=bartpho scripts/pipelines/run_pretrained_seq2seq.sh
```

```bash
MODEL_KEY=mt5 scripts/pipelines/run_pretrained_seq2seq.sh
```

```bash
MODEL_KEY=mbart scripts/pipelines/run_pretrained_seq2seq.sh
```

For H100 MIG environments, the pretrained pipeline defaults to `PIN_MEMORY=0`
and `PERSISTENT_WORKERS=0` to avoid PyTorch/NVML allocator crashes. It also uses
`SEQ2SEQ_BATCH_SIZE=8` with `SEQ2SEQ_GRAD_ACCUM_STEPS=2` by default, which keeps
the effective batch size at 16 without requiring a 16-sample forward pass. If
the run is stable and you want more input pipeline throughput, you can override
the worker settings:

```bash
MODEL_KEY=vit5 PIN_MEMORY=1 PERSISTENT_WORKERS=1 \
  scripts/pipelines/run_pretrained_seq2seq.sh
```

The pretrained pipeline writes reusable processed-row and tokenized-tensor
caches under `cache/seq2seq` by default. Later runs with the same dataset,
model/tokenizer, language settings, and max lengths load those cache files
instead of preprocessing and tokenizing again. To force a rebuild:

```bash
MODEL_KEY=vit5 SEQ2SEQ_REBUILD_CACHE=1 \
  scripts/pipelines/run_pretrained_seq2seq.sh
```

To disable disk cache:

```bash
MODEL_KEY=vit5 SEQ2SEQ_DISK_CACHE=0 \
  scripts/pipelines/run_pretrained_seq2seq.sh
```

Default model IDs:

- `vit5`: `VietAI/vit5-base`
- `bartpho`: `vinai/bartpho-syllable`
- `mt5`: `google/mt5-base`
- `mbart`: `facebook/mbart-large-50-many-to-many-mmt`

Override any model with `MODEL_NAME`, for example:

```bash
MODEL_KEY=vit5 MODEL_NAME=VietAI/vit5-large \
  scripts/pipelines/run_pretrained_seq2seq.sh
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

Pretrained seq2seq outputs use the model key, for example:

- `outputs/vit5_predictions.jsonl`
- `outputs/bartpho_predictions.jsonl`
- `outputs/mt5_predictions.jsonl`
- `outputs/mbart_predictions.jsonl`
