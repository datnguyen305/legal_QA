"""Multi-passage reader with Cross-Passage Answer Verification.

The model mirrors the paper's three scoring signals: pointer-network boundary
probabilities over all passages, content probabilities for answer-token
membership, and cross-passage answer-candidate verification by attention among
candidate representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CrossPassageOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    content_logits: Any
    verification_logits: Any


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


def _missing_deps() -> None:
    if torch is None:
        raise SystemExit("Cross-Passage requires PyTorch. Install dependencies from requirements-models.txt.")


class CrossPassageAnswerVerification(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, hidden: int = 128, dropout: float = 0.1) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.q_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.p_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.match_encoder = nn.LSTM(hidden * 4, hidden // 2, batch_first=True, bidirectional=True)
        self.ptr = nn.LSTMCell(hidden, hidden)
        self.ptr_proj = nn.Linear(hidden * 2, 1)
        self.content_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.verify_proj = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def _masked_mean(self, x: Any, mask: Any) -> Any:
        return x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _att_flow(self, p: Any, q: Any, p_mask: Any, q_mask: Any) -> Any:
        sim = torch.bmm(p, q.transpose(1, 2)).masked_fill(~q_mask.unsqueeze(1), -1e4)
        p2q = torch.bmm(torch.softmax(sim, dim=-1), q)
        q2p_scores = sim.masked_fill(~p_mask.unsqueeze(-1), -1e4).max(dim=2).values
        q2p = torch.bmm(torch.softmax(q2p_scores, dim=-1).unsqueeze(1), p).expand(-1, p.size(1), -1)
        return torch.cat([p, p2q, p * p2q, p * q2p], dim=-1)

    def forward(
        self,
        passage_ids: Any | None = None,
        question_ids: Any | None = None,
        context_ids: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        content_labels: Any | None = None,
    ) -> CrossPassageOutput:
        if passage_ids is None:
            passage_ids = context_ids.unsqueeze(1)
        bsz, passages, plen = passage_ids.size()
        flat = passage_ids.reshape(bsz * passages, plen)
        p_mask = flat != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        q, _ = self.q_encoder(self.embedding(question_ids))
        q_rep = q.unsqueeze(1).expand(bsz, passages, -1, -1).reshape(bsz * passages, q.size(1), q.size(2))
        q_mask_rep = q_mask.unsqueeze(1).expand(bsz, passages, -1).reshape(bsz * passages, q_mask.size(1))
        p, _ = self.p_encoder(self.embedding(flat))
        matched, _ = self.match_encoder(self.dropout(self._att_flow(p, q_rep, p_mask, q_mask_rep)))
        flat_mask = p_mask
        h = self._masked_mean(q_rep, q_mask_rep)
        c = torch.zeros_like(h)
        pointer_logits = []
        for _ in range(2):
            scores = self.ptr_proj(torch.cat([matched, h.unsqueeze(1).expand_as(matched)], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~flat_mask, -1e4)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(attn.unsqueeze(1), matched).squeeze(1)
            h, c = self.ptr(ctx, (h, c))
            pointer_logits.append(scores.reshape(bsz, passages * plen))
        start_logits, end_logits = pointer_logits
        content_logits = self.content_proj(matched).squeeze(-1).reshape(bsz, passages, plen)
        content_probs = torch.sigmoid(content_logits).masked_fill(~(passage_ids != self.pad_token_id), 0.0)
        answer_reps = (content_probs.unsqueeze(-1) * matched.reshape(bsz, passages, plen, -1)).sum(dim=2)
        answer_reps = answer_reps / content_probs.sum(dim=2, keepdim=True).clamp(min=1e-6)
        sim = torch.bmm(answer_reps, answer_reps.transpose(1, 2))
        eye = torch.eye(passages, dtype=torch.bool, device=sim.device).unsqueeze(0)
        sim = sim.masked_fill(eye, -1e4)
        support = torch.bmm(torch.softmax(sim, dim=-1), answer_reps)
        verification_logits = self.verify_proj(torch.cat([answer_reps, support], dim=-1)).squeeze(-1)
        verification_token_logits = verification_logits.unsqueeze(-1).expand(-1, -1, plen).reshape(bsz, passages * plen)
        start_logits = start_logits + verification_token_logits
        end_logits = end_logits + verification_token_logits
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2
            if content_labels is not None:
                content_loss = F.binary_cross_entropy_with_logits(content_logits.reshape(bsz, passages * plen), content_labels.float())
                loss = loss + content_loss
        return CrossPassageOutput(loss, start_logits, end_logits, content_logits, verification_logits)
