"""QANet-style extractive reader.

The implementation follows the paper's proposed components: highway token
projection, stacked encoder blocks made of depthwise separable convolutions,
multi-head self-attention and feed-forward layers, trilinear context-query
attention, shared model encoders, and start/end span prediction from M0/M1/M2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpanOutput:
    loss: Any
    start_logits: Any
    end_logits: Any


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
        raise SystemExit("QANet requires PyTorch. Install dependencies from requirements-models.txt.")


class Highway(nn.Module):
    def __init__(self, size: int, layers: int = 2) -> None:
        _missing_deps()
        super().__init__()
        self.transforms = nn.ModuleList(nn.Linear(size, size) for _ in range(layers))
        self.gates = nn.ModuleList(nn.Linear(size, size) for _ in range(layers))

    def forward(self, x: Any) -> Any:
        for transform, gate in zip(self.transforms, self.gates):
            g = torch.sigmoid(gate(x))
            y = torch.relu(transform(x))
            x = g * y + (1.0 - g) * x
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, hidden: int, kernel_size: int) -> None:
        _missing_deps()
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size, padding=padding, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x: Any) -> Any:
        return self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)


class EncoderBlock(nn.Module):
    """QANet encoder block: conv stack, self-attention, feed-forward."""

    def __init__(self, hidden: int, conv_layers: int, kernel_size: int, heads: int, dropout: float) -> None:
        _missing_deps()
        super().__init__()
        self.convs = nn.ModuleList(DepthwiseSeparableConv(hidden, kernel_size) for _ in range(conv_layers))
        self.conv_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(conv_layers))
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Any, mask: Any) -> Any:
        key_padding_mask = ~mask
        for conv, norm in zip(self.convs, self.conv_norms):
            residual = x
            x = self.dropout(torch.relu(conv(norm(x)))) + residual
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        residual = x
        attn, _ = self.self_attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), key_padding_mask=key_padding_mask)
        x = self.dropout(attn) + residual
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        residual = x
        x = self.dropout(self.ffn(self.ffn_norm(x))) + residual
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class ContextQueryAttention(nn.Module):
    """Trilinear C2Q plus Q2C attention as used by QANet/BiDAF."""

    def __init__(self, hidden: int, dropout: float) -> None:
        _missing_deps()
        super().__init__()
        self.weight = nn.Linear(hidden * 3, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: Any, query: Any, context_mask: Any, query_mask: Any) -> Any:
        bsz, clen, hidden = context.size()
        qlen = query.size(1)
        c = context.unsqueeze(2).expand(bsz, clen, qlen, hidden)
        q = query.unsqueeze(1).expand(bsz, clen, qlen, hidden)
        sim = self.weight(torch.cat([c, q, c * q], dim=-1)).squeeze(-1)
        sim = sim.masked_fill(~query_mask.unsqueeze(1), -1e4)
        c2q = torch.bmm(torch.softmax(sim, dim=-1), query)
        q2c_scores = sim.masked_fill(~context_mask.unsqueeze(-1), -1e4).max(dim=2).values
        q2c = torch.bmm(torch.softmax(q2c_scores, dim=-1).unsqueeze(1), context).expand(-1, clen, -1)
        return self.dropout(torch.cat([context, c2q, context * c2q, context * q2c], dim=-1))


class QANet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden: int = 128,
        heads: int = 8,
        dropout: float = 0.1,
        emb_conv_layers: int = 4,
        model_conv_layers: int = 2,
        model_blocks: int = 7,
        kernel_size: int = 7,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.highway = Highway(hidden)
        self.embed_encoder = EncoderBlock(hidden, emb_conv_layers, kernel_size, heads, dropout)
        self.cq_att = ContextQueryAttention(hidden, dropout)
        self.cq_proj = nn.Linear(hidden * 4, hidden)
        self.model_encoder = nn.ModuleList(
            EncoderBlock(hidden, model_conv_layers, kernel_size, heads, dropout) for _ in range(model_blocks)
        )
        self.start_proj = nn.Linear(hidden * 2, 1)
        self.end_proj = nn.Linear(hidden * 2, 1)

    def _encode_model(self, x: Any, mask: Any) -> Any:
        for block in self.model_encoder:
            x = block(x, mask)
        return x

    def forward(
        self,
        context_ids: Any,
        question_ids: Any,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
    ) -> SpanOutput:
        context_mask = context_ids != self.pad_token_id
        question_mask = question_ids != self.pad_token_id
        c = self.embed_encoder(self.highway(self.embedding(context_ids)), context_mask)
        q = self.embed_encoder(self.highway(self.embedding(question_ids)), question_mask)
        x = self.cq_proj(self.cq_att(c, q, context_mask, question_mask))
        m0 = self._encode_model(x, context_mask)
        m1 = self._encode_model(m0, context_mask)
        m2 = self._encode_model(m1, context_mask)
        start_logits = self.start_proj(torch.cat([m0, m1], dim=-1)).squeeze(-1).masked_fill(~context_mask, -1e4)
        end_logits = self.end_proj(torch.cat([m0, m2], dim=-1)).squeeze(-1).masked_fill(~context_mask, -1e4)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2
        return SpanOutput(loss, start_logits, end_logits)
