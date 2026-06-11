"""S-NET-style extraction-then-synthesis modules.

This implementation follows the original paper's structure more closely than a
pretrained seq2seq baseline:

- evidence is represented as start/end feature indicators on passage tokens;
- the synthesis model uses word embeddings, bidirectional GRU encoders, a GRU
  decoder with attention, and a softmax over a local vocabulary;
- all parameters are trained from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_preprocessing.qa_preprocess import sentence_split, tokenize


@dataclass
class Evidence:
    text: str
    sentence_index: int
    score: float


@dataclass
class SNetOutput:
    loss: Any
    logits: Any


try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


def _missing_deps() -> None:
    if torch is None:
        raise SystemExit("S-NET requires PyTorch. Install dependencies from requirements-models.txt.")


def select_evidence_sentence(question: str, context: str) -> Evidence:
    """Select a sentence as evidence using lexical overlap.

    The original S-NET uses a trained extraction model. This deterministic
    fallback is used at inference when only the synthesis checkpoint is present.
    Training uses gold answer_start/answer_end as evidence labels when present.
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


def token_feature_flags(
    tokens: list[str],
    context: str,
    answer_start: int | None,
    answer_end: int | None,
) -> tuple[list[int], list[int]]:
    """Map character evidence boundaries to token start/end feature flags."""
    start_flags = [0] * len(tokens)
    end_flags = [0] * len(tokens)
    if answer_start is None or answer_end is None or not (0 <= answer_start < answer_end <= len(context)):
        return start_flags, end_flags

    cursor = 0
    evidence_token_indices: list[int] = []
    lower_context = context.lower()
    for i, tok in enumerate(tokens):
        pos = lower_context.find(tok.lower(), cursor)
        if pos < 0:
            pos = cursor
        tok_start = pos
        tok_end = pos + len(tok)
        if tok_end > answer_start and tok_start < answer_end:
            evidence_token_indices.append(i)
        cursor = tok_end
    if evidence_token_indices:
        start_flags[evidence_token_indices[0]] = 1
        end_flags[evidence_token_indices[-1]] = 1
    return start_flags, end_flags


class SNetSynthesis(nn.Module):
    """GRU answer synthesis with evidence start/end feature embeddings."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        embed_size: int = 300,
        feature_size: int = 50,
        hidden_size: int = 150,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.start_feature = nn.Embedding(2, feature_size)
        self.end_feature = nn.Embedding(2, feature_size)
        self.passage_encoder = nn.GRU(
            embed_size + feature_size * 2,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.question_encoder = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_decoder = nn.Linear(hidden_size * 4, hidden_size)
        self.decoder = nn.GRUCell(embed_size + hidden_size * 4, hidden_size)
        self.att_p = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_q = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_d = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_v = nn.Linear(hidden_size, 1, bias=False)
        self.readout = nn.Linear(embed_size + hidden_size + hidden_size * 4, hidden_size)
        self.vocab_proj = nn.Linear(hidden_size, vocab_size)

    def _masked_mean(self, x: Any, mask: Any) -> Any:
        return x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _attend(self, memory: Any, mask: Any, state: Any, proj: Any) -> Any:
        scores = self.att_v(torch.tanh(proj(memory) + self.att_d(state).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), memory).squeeze(1)

    def forward(
        self,
        passage_ids: Any,
        question_ids: Any,
        start_features: Any,
        end_features: Any,
        answer_ids: Any | None = None,
        max_answer_len: int | None = None,
    ) -> SNetOutput:
        passage_mask = passage_ids != self.pad_token_id
        question_mask = question_ids != self.pad_token_id
        passage_input = torch.cat(
            [self.embedding(passage_ids), self.start_feature(start_features), self.end_feature(end_features)],
            dim=-1,
        )
        passage_memory, _ = self.passage_encoder(passage_input)
        question_memory, _ = self.question_encoder(self.embedding(question_ids))
        passage_vec = self._masked_mean(passage_memory, passage_mask)
        question_vec = self._masked_mean(question_memory, question_mask)
        state = torch.tanh(self.init_decoder(torch.cat([passage_vec, question_vec], dim=-1)))
        prev_context = torch.zeros(passage_ids.size(0), passage_memory.size(-1) + question_memory.size(-1), device=passage_ids.device)
        prev = torch.full((passage_ids.size(0),), self.bos_token_id, dtype=torch.long, device=passage_ids.device)

        steps = answer_ids.size(1) - 1 if answer_ids is not None else max_answer_len or 64
        logits = []
        losses = []
        for t in range(steps):
            state = self.decoder(torch.cat([self.embedding(prev), prev_context], dim=-1), state)
            p_context = self._attend(passage_memory, passage_mask, state, self.att_p)
            q_context = self._attend(question_memory, question_mask, state, self.att_q)
            prev_context = torch.cat([p_context, q_context], dim=-1)
            readout = torch.tanh(torch.nn.functional.max_pool1d(
                self.readout(torch.cat([self.embedding(prev), state, prev_context], dim=-1)).unsqueeze(-1),
                kernel_size=1,
            ).squeeze(-1))
            step_logits = self.vocab_proj(readout)
            logits.append(step_logits)
            if answer_ids is not None:
                target = answer_ids[:, t + 1]
                losses.append(torch.nn.functional.cross_entropy(step_logits, target, ignore_index=self.pad_token_id))
                prev = target
            else:
                prev = step_logits.argmax(dim=-1)
        loss = sum(losses) / max(1, len(losses)) if losses else None
        return SNetOutput(loss, torch.stack(logits, dim=1))
