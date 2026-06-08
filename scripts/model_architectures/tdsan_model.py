"""Token-level Dynamic Self-Attention Network for multi-passage MRC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TdsanOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    gate_loss: Any


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
        raise SystemExit("TD-SAN requires: python3 -m pip install -r requirements-models.txt")


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size, padding=padding, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Any) -> Any:
        y = self.norm(x).transpose(1, 2)
        y = self.pointwise(self.depthwise(y)).transpose(1, 2)
        return x + self.dropout(torch.relu(y))


class DynSABlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 8, top_k: int = 256, kernel_size: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.top_k = top_k
        self.local = nn.Sequential(
            DepthwiseSeparableConv(hidden, kernel_size, dropout),
            DepthwiseSeparableConv(hidden, kernel_size, dropout),
        )
        self.gate_u = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, heads)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.last_gate = None

    def forward(self, x: Any, mask: Any | None = None) -> Any:
        u = self.local(x)
        gates = torch.sigmoid(self.gate(torch.relu(self.gate_u(u))))  # B x L x H
        self.last_gate = gates
        token_score = gates.max(dim=-1).values
        if mask is not None:
            token_score = token_score.masked_fill(~mask.bool(), -1e4)
        k = min(self.top_k, x.size(1))
        top_idx = token_score.topk(k, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden)
        chosen = torch.gather(u, 1, gather_idx)
        attended, _ = self.attn(chosen, chosen, chosen, need_weights=False)
        scattered = torch.zeros_like(u).scatter_add(1, gather_idx, attended)
        gate_scale = token_score.clamp(min=0).unsqueeze(-1)
        gate_scale = gate_scale / gate_scale.amax(dim=1, keepdim=True).clamp(min=1e-6)
        y = (self.ff(u) + scattered) * gate_scale
        return self.norm(u + self.dropout(self.out(y)))


class TdsanForQuestionAnswering(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden: int = 128,
        heads: int = 8,
        top_k: int = 256,
        block_layers: int = 2,
        cross_layers: int = 4,
        max_position: int = 4096,
        dropout: float = 0.1,
        gate_beta: float = 1e-5,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.gate_beta = gate_beta
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.position = nn.Embedding(max_position, hidden)
        self.rank = nn.Embedding(64, hidden)
        self.input_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.question_blocks = nn.ModuleList(DynSABlock(hidden, heads, top_k, dropout=dropout) for _ in range(block_layers))
        self.passage_blocks = nn.ModuleList(DynSABlock(hidden, heads, top_k, dropout=dropout) for _ in range(block_layers))
        self.align_proj = nn.Linear(hidden * 4, hidden)
        self.post_align = nn.ModuleList(DynSABlock(hidden, heads, top_k, dropout=dropout) for _ in range(block_layers))
        self.cross_blocks = nn.ModuleList(DynSABlock(hidden, heads, top_k, dropout=dropout) for _ in range(cross_layers))
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)

    def _embed(self, ids: Any, rank_ids: Any | None = None) -> Any:
        pos = torch.arange(ids.size(1), device=ids.device).clamp(max=self.position.num_embeddings - 1)
        x = self.embedding(ids) + self.position(pos).unsqueeze(0)
        if rank_ids is not None:
            x = x + self.rank(rank_ids.clamp(max=self.rank.num_embeddings - 1))
        return self.input_proj(x)

    def _blocks(self, x: Any, mask: Any, blocks: Any) -> Any:
        for block in blocks:
            x = block(x, mask)
        return x

    def _gate_regularization(self) -> Any:
        losses = []
        for module in self.modules():
            if isinstance(module, DynSABlock) and module.last_gate is not None:
                losses.append(module.last_gate.abs().mean())
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gate_beta * sum(losses)

    def forward(
        self,
        question_ids: Any,
        passage_ids: Any,
        passage_rank_ids: Any,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
    ) -> TdsanOutput:
        q_mask = question_ids != self.pad_token_id
        p_mask = passage_ids != self.pad_token_id
        q = self._blocks(self._embed(question_ids), q_mask, self.question_blocks)
        p = self._blocks(self._embed(passage_ids, passage_rank_ids), p_mask, self.passage_blocks)

        attn = torch.bmm(p, q.transpose(1, 2)).masked_fill(~q_mask.unsqueeze(1), -1e4)
        p2q = torch.bmm(torch.softmax(attn, dim=-1), q)
        aligned = self.align_proj(torch.cat([p, p2q, p * p2q, p - p2q], dim=-1))
        x = self._blocks(aligned, p_mask, self.post_align)
        x = self._blocks(x, p_mask, self.cross_blocks)
        start_logits = self.start_head(x).squeeze(-1).masked_fill(~p_mask, -1e4)
        end_logits = self.end_head(x).squeeze(-1).masked_fill(~p_mask, -1e4)
        gate_loss = self._gate_regularization()
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(start_logits, start_positions)
                + torch.nn.functional.cross_entropy(end_logits, end_positions)
            ) + gate_loss
        return TdsanOutput(loss, start_logits, end_logits, gate_loss)
