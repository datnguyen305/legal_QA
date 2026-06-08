"""Deep Cascade Model for multi-document reading comprehension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeepCascadeOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    doc_logits: Any
    para_logits: Any


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
        raise SystemExit("Deep Cascade requires: python3 -m pip install -r requirements-models.txt")


class FeatureRanker(nn.Module):
    def __init__(self, feature_size: int = 5) -> None:
        _missing_deps()
        super().__init__()
        self.linear = nn.Linear(feature_size, 1)

    def forward(self, features: Any) -> Any:
        return self.linear(features).squeeze(-1)


class DeepCascadeReader(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden: int = 128,
        feature_size: int = 5,
        doc_loss_weight: float = 0.5,
        para_loss_weight: float = 0.5,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.doc_loss_weight = doc_loss_weight
        self.para_loss_weight = para_loss_weight
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.q_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.c_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.align_proj = nn.Linear(hidden, hidden)
        self.fuse = nn.Linear(hidden * 4, hidden)
        self.shared_lstm = nn.LSTM(hidden + hidden + feature_size, hidden // 2, batch_first=True, bidirectional=True)
        self.doc_head = nn.Linear(hidden + feature_size, 1)
        self.para_head = nn.Linear(hidden + feature_size, 1)
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)

    def _masked_mean(self, x: Any, mask: Any) -> Any:
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _span_pool(self, x: Any, spans: Any) -> Any:
        rows = []
        for b in range(x.size(0)):
            row = []
            for start, end in spans[b].tolist():
                if start < 0 or end < start:
                    row.append(torch.zeros(x.size(-1), dtype=x.dtype, device=x.device))
                else:
                    row.append(x[b, start : end + 1].mean(dim=0))
            rows.append(torch.stack(row))
        return torch.stack(rows)

    def forward(
        self,
        question_ids: Any,
        context_ids: Any,
        token_features: Any,
        doc_spans: Any,
        para_spans: Any,
        doc_features: Any,
        para_features: Any,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        doc_labels: Any | None = None,
        para_labels: Any | None = None,
    ) -> DeepCascadeOutput:
        q_mask = question_ids != self.pad_token_id
        c_mask = context_ids != self.pad_token_id
        q_encoded, _ = self.q_encoder(self.embedding(question_ids))
        c_encoded, _ = self.c_encoder(self.embedding(context_ids))
        q_vec = self._masked_mean(q_encoded, q_mask)

        scores = torch.bmm(self.align_proj(c_encoded), self.align_proj(q_encoded).transpose(1, 2))
        scores = scores.masked_fill(~q_mask.unsqueeze(1), -1e4)
        attended_q = torch.bmm(torch.softmax(scores, dim=-1), q_encoded)
        fused = torch.relu(self.fuse(torch.cat([c_encoded, attended_q, c_encoded * attended_q, c_encoded - attended_q], dim=-1)))
        shared_in = torch.cat([fused, q_vec.unsqueeze(1).expand(-1, fused.size(1), -1), token_features], dim=-1)
        shared, _ = self.shared_lstm(shared_in)
        shared = shared.masked_fill(~c_mask.unsqueeze(-1), 0.0)

        doc_repr = self._span_pool(shared, doc_spans)
        para_flat = para_spans.reshape(para_spans.size(0), -1, 2)
        para_repr = self._span_pool(shared, para_flat).reshape(para_spans.size(0), para_spans.size(1), para_spans.size(2), -1)
        doc_logits = self.doc_head(torch.cat([doc_repr, doc_features], dim=-1)).squeeze(-1)
        para_logits = self.para_head(torch.cat([para_repr, para_features], dim=-1)).squeeze(-1)
        start_logits = self.start_head(shared).squeeze(-1).masked_fill(~c_mask, -1e4)
        end_logits = self.end_head(shared).squeeze(-1).masked_fill(~c_mask, -1e4)

        losses = []
        if start_positions is not None and end_positions is not None:
            losses.append(
                0.5
                * (
                    torch.nn.functional.cross_entropy(start_logits, start_positions)
                    + torch.nn.functional.cross_entropy(end_logits, end_positions)
                )
            )
        if doc_labels is not None:
            losses.append(self.doc_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, doc_labels.float()))
        if para_labels is not None:
            losses.append(self.para_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(para_logits, para_labels.float()))
        loss = sum(losses) if losses else None
        return DeepCascadeOutput(loss, start_logits, end_logits, doc_logits, para_logits)
