"""FETSF-MRC model implementation.

This follows the proposed model's main components:
- shared pre-trained Transformer encoder,
- scanning module for answer type and evidence sentence prediction,
- evidence-weighted detailed reading module for answer span prediction,
- feedback loss from span logits back to sentence-level scanning attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FetsfOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    type_logits: Any
    evidence_logits: Any


def require_torch_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("FETSF-MRC requires: python3 -m pip install -r requirements-models.txt") from exc
    return torch, nn, AutoModel


try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel
except ImportError:
    torch = None
    AutoModel = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


class ScanningTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
            average_attn_weights=False,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Any, key_padding_mask: Any | None = None) -> tuple[Any, Any]:
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class FetsfMRC(nn.Module):
    def __init__(
        self,
        base_model: str,
        max_sentences: int = 64,
        num_answer_types: int = 3,
        dropout: float = 0.1,
        feedback_temperature: float = 2.0,
    ) -> None:
        if torch is None or AutoModel is None:
            require_torch_transformers()
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        heads = getattr(self.encoder.config, "num_attention_heads", 8)
        self.max_sentences = max_sentences
        self.feedback_temperature = feedback_temperature

        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, num_answer_types)
        self.scanning_layers = nn.ModuleList(
            [ScanningTransformerLayer(hidden, heads, dropout), ScanningTransformerLayer(hidden, heads, dropout)]
        )
        self.evidence_head = nn.Linear(hidden, 1)
        self.reader = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)

    def _pool_sentences(self, hidden: Any, sentence_spans: Any) -> tuple[Any, Any]:
        batch, sentence_count, _ = sentence_spans.shape
        pooled = []
        mask = sentence_spans[:, :, 0] < 0
        for b in range(batch):
            row = []
            for j in range(sentence_count):
                start, end = sentence_spans[b, j].tolist()
                if start < 0 or end < start:
                    row.append(torch.zeros(hidden.size(-1), device=hidden.device, dtype=hidden.dtype))
                else:
                    row.append(hidden[b, start : end + 1].max(dim=0).values)
            pooled.append(torch.stack(row))
        return torch.stack(pooled), mask

    def _token_evidence_weights(self, evidence_logits: Any, sentence_spans: Any, seq_len: int) -> Any:
        weights = torch.zeros(
            evidence_logits.size(0),
            seq_len,
            device=evidence_logits.device,
            dtype=evidence_logits.dtype,
        )
        scores = torch.sigmoid(evidence_logits)
        for b in range(sentence_spans.size(0)):
            for j in range(sentence_spans.size(1)):
                start, end = sentence_spans[b, j].tolist()
                if start >= 0 and end >= start:
                    weights[b, start : end + 1] = scores[b, j]
        return weights

    def _feedback_loss(self, start_logits: Any, end_logits: Any, sentence_spans: Any, attn_weights: Any) -> Any:
        sentence_scores = []
        for b in range(sentence_spans.size(0)):
            scores = []
            for j in range(sentence_spans.size(1)):
                start, end = sentence_spans[b, j].tolist()
                if start < 0 or end < start:
                    scores.append(torch.tensor(-1e4, device=start_logits.device, dtype=start_logits.dtype))
                else:
                    scores.append(start_logits[b, start : end + 1].max() + end_logits[b, start : end + 1].max())
            sentence_scores.append(torch.stack(scores))
        alpha = torch.stack(sentence_scores)

        beta = attn_weights.mean(dim=1).mean(dim=1)
        valid = sentence_spans[:, :, 0] >= 0
        alpha = alpha.masked_fill(~valid, -1e4)
        beta = beta.masked_fill(~valid, -1e4)
        p = torch.softmax(alpha / self.feedback_temperature, dim=-1)
        q_log = torch.log_softmax(beta / self.feedback_temperature, dim=-1)
        return torch.nn.functional.kl_div(q_log, p, reduction="batchmean")

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
        sentence_spans: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        evidence_labels: Any | None = None,
        answer_type: Any | None = None,
    ) -> FetsfOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = self.dropout(encoded.last_hidden_state)
        doc_repr = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4).max(dim=1).values
        type_logits = self.type_head(doc_repr)

        if sentence_spans is None:
            sentence_spans = torch.full(
                (input_ids.size(0), 1, 2),
                -1,
                device=input_ids.device,
                dtype=torch.long,
            )
        sent_repr, sent_padding = self._pool_sentences(hidden, sentence_spans)
        attn_weights = None
        for layer in self.scanning_layers:
            sent_repr, attn_weights = layer(sent_repr, key_padding_mask=sent_padding)
        evidence_logits = self.evidence_head(sent_repr).squeeze(-1).masked_fill(sent_padding, -1e4)

        token_weights = self._token_evidence_weights(evidence_logits, sentence_spans, input_ids.size(1))
        weighted_hidden = hidden * (1.0 + token_weights.unsqueeze(-1))
        reader_hidden = self.reader(weighted_hidden, src_key_padding_mask=~attention_mask.bool())
        start_logits = self.start_head(reader_hidden).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        end_logits = self.end_head(reader_hidden).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)

        losses = []
        if answer_type is not None:
            losses.append(torch.nn.functional.cross_entropy(type_logits, answer_type))
        if evidence_labels is not None:
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    evidence_logits,
                    evidence_labels.float(),
                    reduction="none",
                )
                .masked_fill(sent_padding, 0.0)
                .sum()
                / sent_padding.logical_not().sum().clamp(min=1)
            )
        if start_positions is not None and end_positions is not None:
            losses.append(torch.nn.functional.cross_entropy(start_logits, start_positions))
            losses.append(torch.nn.functional.cross_entropy(end_logits, end_positions))
            if attn_weights is not None:
                losses.append(self._feedback_loss(start_logits, end_logits, sentence_spans, attn_weights))
        loss = sum(losses) if losses else None
        return FetsfOutput(loss, start_logits, end_logits, type_logits, evidence_logits)
