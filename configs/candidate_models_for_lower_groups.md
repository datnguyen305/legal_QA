# Candidate Models for the Smaller Comparison Groups

This note lists additional paper models that can make the smaller comparison
groups less under-populated. These are candidates only; they are not implemented
in this repo yet.

## Group: Custom Neural Extractive Readers

These models are closer to `TD-SAN` and `Deep Cascade` than to pretrained
Transformer readers, because they are classic neural MRC architectures that do
not require BERT/RoBERTa-style pretrained encoders.

| Candidate | Fit | Why it fits |
|---|---|---|
| Cross-Passage Answer Verification | Strong | Multi-passage MRC model with answer boundary prediction, answer content modeling, and cross-passage answer verification. Good match for multi-document/multi-passage legal QA. |
| QANet | Strong | Extractive span reader built from convolution and self-attention rather than recurrent networks or pretrained Transformers. Good from-scratch comparison for TD-SAN. |
| BiDAF | Medium | Classic extractive reader with bidirectional attention flow. Single-passage by default, but simple and useful as a from-scratch baseline. |
| Match-LSTM + Answer Pointer | Medium | Classic extractive span reader with answer-pointer decoding. Single-passage by default, but relevant as a non-pretrained pointer-style extractor. |
| R-Net | Medium | Gated attention/self-matching extractive reader. Single-passage by default, but historically strong and useful as an older neural baseline. |
| Reinforced Mnemonic Reader | Medium | Enhanced attentive reader with reinforcement-style training. Stronger but more complex to implement cleanly. |

Recommended first addition: **Cross-Passage Answer Verification**, because it is
multi-passage and closest to the legal QA setting.

Recommended second addition: **QANet**, because it is from-scratch,
self-attention based, and a natural comparator to TD-SAN.

## Group: Custom Generative / Copy-Augmented Readers

These models are closer to `CPG` because they generate answer text or synthesize
answers from extracted evidence instead of only selecting a single span.

| Candidate | Fit | Why it fits |
|---|---|---|
| S-Net | Strong | Extraction-then-synthesis framework for MRC. It extracts evidence and then generates/synthesizes the final answer. Good comparator to pointer-generator CPG. |
| LatentQA | Strong | Generates well-formed answers with a stochastic selector network using words from the question, paragraph, and global vocabulary. |
| Multi-span Style Extraction for Generative Reading Comprehension | Medium | Builds generative-style answers from multiple extracted spans. Good for answers that are not one continuous context span. |
| Composing Answer from Multi-spans for Reading Comprehension | Medium | Composes answers from high-confidence extracted n-gram/multi-span candidates. |

Recommended first addition: **S-Net**, because it is explicitly designed for
MRC answer generation from extracted evidence and multi-passage MS-MARCO-style
settings.

Recommended second addition: **LatentQA**, because it is a clean generative QA
architecture and gives CPG another generative comparator.

## Practical Implementation Order

1. Cross-Passage Answer Verification
2. QANet
3. S-Net
4. LatentQA

This order improves the smallest comparison groups while keeping implementation
risk manageable.

