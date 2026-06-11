"""S-NET helpers: extraction features for answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from data_preprocessing.qa_preprocess import sentence_split, tokenize


@dataclass
class Evidence:
    text: str
    sentence_index: int
    score: float


def select_evidence_sentence(question: str, context: str) -> Evidence:
    """Select a sentence as evidence using question/context lexical overlap.

    S-NET's first stage predicts evidence spans. At inference time this local
    implementation uses a deterministic extractor so the synthesis model can run
    without a separately trained span/ranking checkpoint.
    """
    sentences = sentence_split(context)
    q_tokens = set(tokenize(question))
    best = Evidence("", -1, 0.0)
    for idx, sentence in enumerate(sentences):
        s_tokens = set(tokenize(sentence))
        overlap = len(q_tokens & s_tokens)
        score = overlap / max(1, len(q_tokens)) + overlap / max(1, len(s_tokens))
        if score > best.score:
            best = Evidence(sentence, idx, score)
    if best.text:
        return best
    return Evidence(context[:512], 0 if context else -1, 0.0)


def snet_input(question: str, context: str, evidence: str) -> str:
    return f"question: {question} evidence: {evidence} passage: {context}"
