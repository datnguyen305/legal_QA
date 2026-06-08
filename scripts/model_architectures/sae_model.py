"""Select, Answer and Explain (SAE) model implementation.

Implements the proposed AAAI-20 SAE architecture:
- document selection with BERT CLS embeddings, multi-head self-attention, and
  pairwise learning-to-rank loss,
- answer/explain module with answer span heads,
- mixed attentive pooling for sentence embeddings,
- sentence-level GNN explanation classifier,
- answer type classifier over graph representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SaeSelectorOutput:
    loss: Any
    doc_scores: Any
    pair_logits: Any


@dataclass
class SaeOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    support_logits: Any
    answer_type_logits: Any


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


def _missing_deps() -> None:
    if torch is None or AutoModel is None:
        raise SystemExit("SAE requires: python3 -m pip install -r requirements-models.txt")


class SaeDocumentSelector(nn.Module):
    def __init__(self, base_model: str, max_docs: int = 8, dropout: float = 0.1) -> None:
        _missing_deps()
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        heads = getattr(self.encoder.config, "num_attention_heads", 8)
        self.max_docs = max_docs
        self.doc_attention = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=dropout,
            batch_first=True,
            average_attn_weights=False,
        )
        self.bilinear = nn.Bilinear(hidden, hidden, 1)
        self.doc_head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
        doc_labels: Any | None = None,
        doc_scores: Any | None = None,
    ) -> SaeSelectorOutput:
        batch, docs, seq_len = input_ids.shape
        flat = {
            "input_ids": input_ids.reshape(batch * docs, seq_len),
            "attention_mask": attention_mask.reshape(batch * docs, seq_len),
        }
        if token_type_ids is not None:
            flat["token_type_ids"] = token_type_ids.reshape(batch * docs, seq_len)
        encoded = self.encoder(**flat)
        cls = encoded.last_hidden_state[:, 0].reshape(batch, docs, -1)
        doc_mask = attention_mask.sum(dim=-1) == 0
        attended, _ = self.doc_attention(cls, cls, cls, key_padding_mask=doc_mask)
        attended = self.dropout(attended).masked_fill(doc_mask.unsqueeze(-1), 0.0)

        pair_logits = []
        for i in range(docs):
            row = []
            for j in range(docs):
                row.append(self.bilinear(attended[:, i], attended[:, j]).squeeze(-1))
            pair_logits.append(torch.stack(row, dim=1))
        pair_logits = torch.stack(pair_logits, dim=1)
        relevance = (pair_logits > 0).float().sum(dim=-1).masked_fill(doc_mask, -1e4)

        loss = None
        if doc_scores is not None:
            pair_targets = (doc_scores.unsqueeze(2) > doc_scores.unsqueeze(1)).float()
            valid = (~doc_mask).unsqueeze(2) & (~doc_mask).unsqueeze(1)
            eye = torch.eye(docs, dtype=torch.bool, device=input_ids.device).unsqueeze(0)
            valid = valid & ~eye
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                pair_logits,
                pair_targets,
                reduction="none",
            )
            pair_loss = raw.masked_select(valid).mean() if valid.any() else raw.mean()
            losses = [pair_loss]
            if doc_labels is not None:
                point_logits = self.doc_head(attended).squeeze(-1).masked_fill(doc_mask, -1e4)
                point_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    point_logits,
                    doc_labels.float(),
                    reduction="none",
                )
                losses.append(point_raw.masked_select(~doc_mask).mean())
            loss = sum(losses)
        return SaeSelectorOutput(loss, relevance, pair_logits)


class RelationalGnnLayer(nn.Module):
    def __init__(self, hidden: int, relation_count: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden, hidden)
        self.rel_proj = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(relation_count))
        self.gate = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Any, adjacency: Any, node_mask: Any) -> Any:
        update = self.self_proj(h)
        for rel_id, proj in enumerate(self.rel_proj):
            adj = adjacency[:, rel_id]
            denom = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            msg = torch.bmm(adj, proj(h)) / denom
            update = update + msg
        update = torch.relu(update)
        gate = torch.sigmoid(self.gate(torch.cat([update, h], dim=-1)))
        h_next = gate * update + (1.0 - gate) * h
        return self.dropout(h_next).masked_fill(~node_mask.unsqueeze(-1), 0.0)


class SaeAnswerExplain(nn.Module):
    def __init__(
        self,
        base_model: str,
        max_sentences: int = 96,
        gnn_layers: int = 2,
        relation_count: int = 3,
        span_loss_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.max_sentences = max_sentences
        self.span_loss_weight = span_loss_weight
        self.span_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.token_attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.gnn = nn.ModuleList(RelationalGnnLayer(hidden, relation_count, dropout) for _ in range(gnn_layers))
        self.support_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.answer_type_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.dropout = nn.Dropout(dropout)

    def _sentence_embeddings(self, hidden: Any, span_logits: Any, sentence_spans: Any) -> tuple[Any, Any]:
        batch, sent_count, _ = sentence_spans.shape
        embeddings = []
        mask = sentence_spans[:, :, 0] >= 0
        for b in range(batch):
            row = []
            for j in range(sent_count):
                start, end = sentence_spans[b, j].tolist()
                if start < 0 or end < start:
                    row.append(torch.zeros(hidden.size(-1), dtype=hidden.dtype, device=hidden.device))
                    continue
                tokens = hidden[b, start : end + 1]
                logits = span_logits[b, start : end + 1]
                mixed = self.token_attn(tokens).squeeze(-1) + logits[:, 0] + logits[:, 1]
                weights = torch.softmax(mixed, dim=0)
                row.append((weights.unsqueeze(-1) * tokens).sum(dim=0))
            embeddings.append(torch.stack(row))
        return torch.stack(embeddings), mask

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
        sentence_spans: Any | None = None,
        adjacency: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        support_labels: Any | None = None,
        answer_type: Any | None = None,
    ) -> SaeOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = self.dropout(encoded.last_hidden_state)
        span_logits = self.span_head(hidden).masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4)
        start_logits = span_logits[:, :, 0]
        end_logits = span_logits[:, :, 1]

        if sentence_spans is None:
            sentence_spans = torch.full((input_ids.size(0), 1, 2), -1, dtype=torch.long, device=input_ids.device)
        sent_repr, node_mask = self._sentence_embeddings(hidden, span_logits, sentence_spans)
        if adjacency is None:
            adjacency = torch.zeros(
                input_ids.size(0),
                3,
                sentence_spans.size(1),
                sentence_spans.size(1),
                dtype=hidden.dtype,
                device=input_ids.device,
            )
        for layer in self.gnn:
            sent_repr = layer(sent_repr, adjacency.float(), node_mask)
        support_logits = self.support_head(sent_repr).squeeze(-1).masked_fill(~node_mask, -1e4)
        support_weights = torch.softmax(support_logits.masked_fill(~node_mask, -1e4), dim=-1)
        graph_repr = (support_weights.unsqueeze(-1) * sent_repr).sum(dim=1)
        answer_type_logits = self.answer_type_head(graph_repr)

        losses = []
        if start_positions is not None and end_positions is not None:
            losses.append(
                self.span_loss_weight
                * 0.5
                * (
                    torch.nn.functional.cross_entropy(start_logits, start_positions)
                    + torch.nn.functional.cross_entropy(end_logits, end_positions)
                )
            )
        if support_labels is not None:
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                support_logits,
                support_labels.float(),
                reduction="none",
            )
            losses.append(raw.masked_select(node_mask).mean() if node_mask.any() else raw.mean())
        if answer_type is not None:
            losses.append(torch.nn.functional.cross_entropy(answer_type_logits, answer_type))
        loss = sum(losses) if losses else None
        return SaeOutput(loss, start_logits, end_logits, support_logits, answer_type_logits)
