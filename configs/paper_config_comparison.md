# Implementation Configs vs Paper Configs

This file compares the default configs in `configs/models/*.json` against the
hyperparameters explicitly reported in the provided paper PDFs. Values marked
`not stated` were not found as fixed settings in the paper text.

## Summary Table

| Model | Paper-reported config | Our implementation config | Match level |
|---|---|---|---|
| EQUALS | Bert-base-Chinese; MRC batch size 4; learning rate 2e-5; max answer length 261; linear warmup 10%; dropout 0.1; Sentence-BERT retrieval batch size 16 and learning rate 2e-5; epochs not clearly stated. | `bert-base-multilingual-cased`; batch size 4; learning rate 2e-5; max answer length 261; max length 512; epochs 2; retriever default `gold`; optional `bm25`/`sbert`. | Partial. Batch size, LR, max answer length match; base model/language and default retriever differ; epochs are repo choice. |
| FETSF-MRC | RoBERTa-base initialized with Chinese-RoBERTa-wwm-ext; learning rate searched in {2e-5, 3e-5}; warmup 0.1; batch size 4; fine-tune 2 epochs on CJRC, 10 on CAIL2020, 5 on HotpotQA. | `bert-base-multilingual-cased`; batch size 4; learning rate 2e-5; epochs 2; max length 512; max sentences 64. | Partial. Batch size, one LR choice, and CJRC epoch count match; encoder differs; search/grid and feedback-temperature settings are simplified. |
| SAE | BERT base uncased and RoBERTa variants; top-2 documents used at evaluation; document selector trained on all samples; answer/explain module trained on gold documents; epochs, LR, and batch size not clearly stated in extracted text. | `bert-base-multilingual-cased`; selector epochs 1; answer epochs 2; batch size 2; learning rate 2e-5; max docs 6; top-k 2. | Approximation. Top-2 evaluation behavior matches; base model differs; several training hyperparameters are repo choices because paper did not clearly state them. |
| TD-SAN / DynSAN | Standard dimension 128; heads 8; chosen tokens K=256; cross-passage DynSA blocks N=4; batch size 32; dropout 0.1; Adam learning rate 0.001; warmup first 500 steps; trained on four 12GB K80 GPUs; epochs not clearly stated. | hidden 128; heads 8; top-k 256; max passages 4; batch size 8; learning rate 0.001; epochs 2. | Partial. Hidden/heads/top-k/LR match; batch size is smaller; dropout/warmup are simplified or implicit; epochs are repo choice. |
| Deep Cascade | K=4 selected documents; N=2 paragraphs; Adam; batch size 32; learning rate 0.0005; GloVe 300d for TriviaQA and word2vec for DuReader; fixed embeddings; LSTM hidden 150 for TriviaQA and 128 for DuReader; epochs not clearly stated. | max docs 4; max paragraphs 2; batch size 8; learning rate 0.0005; hidden 128; reader epochs 2; ranker epochs 1; local token model rather than GloVe/word2vec embedding setup. | Partial. K/N/LR/hidden-for-DuReader match; batch size and embedding setup differ; epochs are repo choice. |
| CPG | Chunk sizes {50, 100, 200, 500}; Adadelta; initial learning rate reported; decoder size 256; block size 200; pretrained GloVe embeddings; max context size and max answer length tuned; epochs symbolic `numEpochs`. | chunk sizes [50, 100, 200, 500]; learning rate 1.0; decoder hidden 256; block size 200; hidden 128; max context tokens 2000; max answer tokens 64; epochs 2. | Partial. Chunk sizes, decoder size, and block size match; GloVe/tuned lengths are simplified; epochs are repo choice. |
| RE3 | BERT-base or BERT-large; Adam learning rate 3e-5; warmup first 10%; fine-tune 2 epochs; batch size 32; dropout 0.1; early-stopped block J=3 for base, J=6 for large; top-5 articles input; max sequence length 384; stride 128; M=20 proposed answers; NMS threshold M*=5. | `bert-base-multilingual-cased`; learning rate 3e-5; epochs 2; batch size 4; early layer 3; max length 384; stride 128; max candidates 5. | Partial. LR/epochs/J/max length/stride match; batch size is smaller; base model differs; proposed-answer/NMS setup is simplified. |

## Main Differences To Report

- The repo uses `bert-base-multilingual-cased` for BERT-style models because the
  dataset is Vietnamese legal QA, while several papers used Chinese or English
  BERT/RoBERTa variants.
- Repo batch sizes are often smaller than paper batch sizes, mostly to make local
  runs feasible.
- Several papers do not clearly state a fixed epoch count; those repo values are
  implementation defaults, not paper-exact settings.
- Oracle/upper-bound evaluation is not a model setting. Repo default evaluation
  is prediction mode: `EVAL_UPPER_BOUND=0`.
- The configs are faithful enough for controlled local experiments, but they are
  not exact paper reproduction configs for every model.

