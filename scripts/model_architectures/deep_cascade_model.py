"""Deep Cascade model for multi-document reading comprehension.

This module implements the paper's coarse-to-fine idea inside one trainable
reader: shared question/document encoding, co-attention and gated fusion,
self-alignment, auxiliary document/paragraph extraction heads, and final answer
span extraction scored together with the coarse extraction heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeepCascadeOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    document_logits: Any
    paragraph_logits: Any


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
        raise SystemExit("Deep Cascade requires PyTorch. Install dependencies from requirements-models.txt.")


class Fusion(nn.Module):
    def __init__(self, hidden: int) -> None:
        _missing_deps()
        super().__init__()
        self.proj = nn.Linear(hidden * 4, hidden)
        self.gate = nn.Linear(hidden * 4, hidden)

    def forward(self, x: Any, y: Any) -> Any:
        z = torch.cat([x, y, x - y, x * y], dim=-1)
        g = torch.sigmoid(self.gate(z))
        return g * torch.relu(self.proj(z)) + (1.0 - g) * x


class DeepCascadeReader(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, hidden: int = 128, dropout: float = 0.1) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.q_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.d_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.align_proj = nn.Linear(hidden, hidden)
        self.fusion = Fusion(hidden)
        self.self_align = nn.Linear(hidden, hidden, bias=False)
        self.doc_query = nn.Linear(hidden, 1)
        self.doc_bilinear = nn.Bilinear(hidden, hidden, 1, bias=False)
        self.para_bilinear = nn.Bilinear(hidden, hidden, 1, bias=False)
        self.start_proj = nn.Linear(hidden * 2, 1)
        self.end_proj = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def _masked_mean(self, x: Any, mask: Any) -> Any:
        return x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _weighted(self, x: Any, mask: Any, scorer: Any) -> Any:
        logits = scorer(x).squeeze(-1).masked_fill(~mask, -1e4)
        return torch.bmm(torch.softmax(logits, dim=-1).unsqueeze(1), x).squeeze(1)

    def forward(
        self,
        passage_ids: Any | None = None,
        question_ids: Any | None = None,
        context_ids: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        document_labels: Any | None = None,
        paragraph_labels: Any | None = None,
    ) -> DeepCascadeOutput:
        if passage_ids is None:
            passage_ids = context_ids.unsqueeze(1)
        bsz, passages, plen = passage_ids.size()
        flat = passage_ids.reshape(bsz * passages, plen)
        p_mask = flat != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        q, _ = self.q_encoder(self.embedding(question_ids))
        d, _ = self.d_encoder(self.embedding(flat))
        q_rep = q.unsqueeze(1).expand(bsz, passages, -1, -1).reshape(bsz * passages, q.size(1), q.size(2))
        q_mask_rep = q_mask.unsqueeze(1).expand(bsz, passages, -1).reshape(bsz * passages, q_mask.size(1))
        sim = torch.bmm(torch.relu(self.align_proj(d)), torch.relu(self.align_proj(q_rep)).transpose(1, 2))
        sim = sim.masked_fill(~q_mask_rep.unsqueeze(1), -1e4)
        attended_q = torch.bmm(torch.softmax(sim, dim=-1), q_rep)
        fused = self.fusion(d, attended_q)
        self_scores = torch.bmm(self.self_align(fused), fused.transpose(1, 2)).masked_fill(~p_mask.unsqueeze(1), -1e4)
        self_ctx = torch.bmm(torch.softmax(self_scores, dim=-1), fused)
        encoded = self.dropout(self.fusion(fused, self_ctx))
        token_doc = encoded.reshape(bsz, passages, plen, -1)
        token_mask = passage_ids != self.pad_token_id
        q_vec = self._weighted(q, q_mask, self.doc_query)
        para_vec = (token_doc * token_mask.unsqueeze(-1)).sum(dim=2) / token_mask.sum(dim=2, keepdim=True).clamp(min=1)
        q_para = q_vec.unsqueeze(1).expand_as(para_vec)
        document_logits = self.doc_bilinear(q_para, para_vec).squeeze(-1)
        paragraph_logits = self.para_bilinear(q_para, para_vec).squeeze(-1)
        coarse = (document_logits + paragraph_logits).unsqueeze(-1).expand(-1, -1, plen).reshape(bsz, passages * plen)
        q_token = q_vec.unsqueeze(1).unsqueeze(1).expand(-1, passages, plen, -1).reshape(bsz, passages * plen, -1)
        flat_encoded = token_doc.reshape(bsz, passages * plen, -1)
        flat_mask = token_mask.reshape(bsz, passages * plen)
        start_logits = self.start_proj(torch.cat([flat_encoded, q_token], dim=-1)).squeeze(-1) + coarse
        end_logits = self.end_proj(torch.cat([flat_encoded, q_token], dim=-1)).squeeze(-1) + coarse
        start_logits = start_logits.masked_fill(~flat_mask, -1e4)
        end_logits = end_logits.masked_fill(~flat_mask, -1e4)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2
            if document_labels is not None:
                loss = loss + F.binary_cross_entropy_with_logits(document_logits, document_labels.float())
            if paragraph_labels is not None:
                loss = loss + F.binary_cross_entropy_with_logits(paragraph_logits, paragraph_labels.float())
        return DeepCascadeOutput(loss, start_logits, end_logits, document_logits, paragraph_logits)
