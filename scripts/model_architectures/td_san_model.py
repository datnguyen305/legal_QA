"""Token-level Dynamic Self-Attention Network (DynSAN / TD-SAN).

The core DynSA block uses local convolution, learned gates to select top-K
tokens per attention head, scaled dot-product attention only among selected
tokens, and a gated residual update. The reader applies DynSA before alignment,
after alignment, and over concatenated passages for token-level cross-passage
attention with optional passage-rank embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TDSANOutput:
    loss: Any
    start_logits: Any
    end_logits: Any
    gates: Any


try:
    import math
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
        raise SystemExit("TD-SAN requires PyTorch. Install dependencies from requirements-models.txt.")


class DynamicSelfAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 8, top_k: int = 64, kernel_size: int = 7, dropout: float = 0.1) -> None:
        _missing_deps()
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.top_k = top_k
        self.head_dim = hidden // heads
        self.local = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.local_point = nn.Conv1d(hidden, hidden, 1)
        self.gate_hidden = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, heads)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.update_gate = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Any, mask: Any) -> tuple[Any, Any]:
        residual = x
        x = torch.relu(self.local_point(self.local(x.transpose(1, 2))).transpose(1, 2))
        x = self.norm(self.dropout(x) + residual)
        gates = torch.sigmoid(self.gate(torch.relu(self.gate_hidden(x)))).transpose(1, 2)
        gates = gates.masked_fill(~mask.unsqueeze(1), -1.0)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        bsz, seq_len, _ = x.size()
        q = q.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        attended = torch.zeros_like(q)
        k_select = min(self.top_k, seq_len)
        valid_count = mask.sum(dim=1).clamp(min=1)
        for h in range(self.heads):
            head_gate = gates[:, h]
            top_idx = head_gate.topk(k_select, dim=-1).indices
            batch_idx = torch.arange(bsz, device=x.device).unsqueeze(-1)
            qh = q[:, h][batch_idx, top_idx]
            kh = k[:, h][batch_idx, top_idx]
            vh = v[:, h][batch_idx, top_idx]
            scores = torch.bmm(qh, kh.transpose(1, 2)) / math.sqrt(self.head_dim)
            selected_valid = torch.arange(k_select, device=x.device).unsqueeze(0) < valid_count.unsqueeze(1).clamp(max=k_select)
            scores = scores.masked_fill(~selected_valid.unsqueeze(1), -1e4)
            ah = torch.bmm(torch.softmax(scores, dim=-1), vh)
            head_out = torch.zeros(bsz, seq_len, self.head_dim, dtype=x.dtype, device=x.device)
            head_out.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, self.head_dim), ah.to(head_out.dtype))
            attended[:, h] = head_out
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, self.hidden)
        candidate = self.out(attended)
        update = torch.sigmoid(self.update_gate(torch.cat([x, candidate], dim=-1)))
        y = update * candidate + (1.0 - update) * x
        return y.masked_fill(~mask.unsqueeze(-1), 0.0), gates


class TDSANReader(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden: int = 128,
        heads: int = 8,
        top_k: int = 64,
        local_layers: int = 1,
        cross_layers: int = 2,
        max_passages: int = 16,
        dropout: float = 0.1,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.rank_embedding = nn.Embedding(max_passages, hidden)
        self.local_blocks = nn.ModuleList(DynamicSelfAttentionBlock(hidden, heads, top_k, dropout=dropout) for _ in range(local_layers))
        self.post_align_blocks = nn.ModuleList(DynamicSelfAttentionBlock(hidden, heads, top_k, dropout=dropout) for _ in range(local_layers))
        self.cross_blocks = nn.ModuleList(DynamicSelfAttentionBlock(hidden, heads, top_k, dropout=dropout) for _ in range(cross_layers))
        self.trilinear = nn.Linear(hidden * 3, 1, bias=False)
        self.align_proj = nn.Linear(hidden * 4, hidden)
        self.start_proj = nn.Linear(hidden, 1)
        self.end_proj = nn.Linear(hidden, 1)

    def _dyn_stack(self, x: Any, mask: Any, blocks: Any) -> tuple[Any, list[Any]]:
        gates = []
        for block in blocks:
            x, gate = block(x, mask)
            gates.append(gate)
        return x, gates

    def _align(self, p: Any, q: Any, p_mask: Any, q_mask: Any) -> Any:
        bsz, plen, hidden = p.size()
        qlen = q.size(1)
        pp = p.unsqueeze(2).expand(bsz, plen, qlen, hidden)
        qq = q.unsqueeze(1).expand(bsz, plen, qlen, hidden)
        sim = self.trilinear(torch.cat([pp, qq, pp * qq], dim=-1)).squeeze(-1)
        sim = sim.masked_fill(~q_mask.unsqueeze(1), -1e4)
        p2q = torch.bmm(torch.softmax(sim, dim=-1), q)
        q2p_scores = sim.masked_fill(~p_mask.unsqueeze(-1), -1e4).max(dim=2).values
        q2p = torch.bmm(torch.softmax(q2p_scores, dim=-1).unsqueeze(1), p).expand(-1, plen, -1)
        return self.align_proj(torch.cat([p, p2q, p * p2q, p * q2p], dim=-1))

    def forward(
        self,
        passage_ids: Any | None = None,
        question_ids: Any | None = None,
        context_ids: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
    ) -> TDSANOutput:
        if passage_ids is None:
            passage_ids = context_ids.unsqueeze(1)
        bsz, passages, plen = passage_ids.size()
        flat = passage_ids.reshape(bsz * passages, plen)
        p_mask = flat != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        q = self.embedding(question_ids)
        p = self.embedding(flat)
        q, q_gates = self._dyn_stack(q, q_mask, self.local_blocks)
        p, p_gates = self._dyn_stack(p, p_mask, self.local_blocks)
        q_rep = q.unsqueeze(1).expand(bsz, passages, -1, -1).reshape(bsz * passages, q.size(1), q.size(2))
        q_mask_rep = q_mask.unsqueeze(1).expand(bsz, passages, -1).reshape(bsz * passages, q_mask.size(1))
        aligned = self._align(p, q_rep, p_mask, q_mask_rep)
        aligned, align_gates = self._dyn_stack(aligned, p_mask, self.post_align_blocks)
        passage_seq = aligned.reshape(bsz, passages, plen, -1)
        ranks = torch.arange(passages, device=passage_ids.device).clamp(max=self.rank_embedding.num_embeddings - 1)
        passage_seq = passage_seq + self.rank_embedding(ranks).view(1, passages, 1, -1)
        cross = passage_seq.reshape(bsz, passages * plen, -1)
        cross_mask = (passage_ids != self.pad_token_id).reshape(bsz, passages * plen)
        cross, cross_gates = self._dyn_stack(cross, cross_mask, self.cross_blocks)
        start_logits = self.start_proj(cross).squeeze(-1).masked_fill(~cross_mask, -1e4)
        end_logits = self.end_proj(cross).squeeze(-1).masked_fill(~cross_mask, -1e4)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2
        return TDSANOutput(loss, start_logits, end_logits, q_gates + p_gates + align_gates + cross_gates)
