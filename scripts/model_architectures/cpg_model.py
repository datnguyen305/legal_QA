"""Curriculum Pointer-Generator with Introspective Alignment Layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CpgOutput:
    loss: Any
    logits: Any
    copy_attention: Any


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
        raise SystemExit("Curriculum pointer-generator requires: python3 -m pip install -r requirements-models.txt")


class IntrospectiveAlignmentLayer(nn.Module):
    def __init__(self, hidden: int, block_size: int = 200) -> None:
        _missing_deps()
        super().__init__()
        self.block_size = block_size
        self.proj = nn.Linear(hidden, hidden)
        self.self_proj = nn.Linear(hidden * 4, hidden)
        self.out = nn.LSTM(hidden * 4, hidden // 2, batch_first=True, bidirectional=True)

    def forward(self, context: Any, question: Any, question_mask: Any) -> Any:
        c = torch.relu(self.proj(context))
        q = torch.relu(self.proj(question))
        affinity = torch.bmm(c, q.transpose(1, 2)).masked_fill(~question_mask.unsqueeze(1), -1e4)
        aligned = torch.bmm(torch.softmax(affinity, dim=-1), question)
        decomp = torch.cat([aligned, context, aligned - context, aligned * context], dim=-1)
        local = torch.zeros_like(context)
        for start in range(0, context.size(1), self.block_size):
            end = min(start + self.block_size, context.size(1))
            block = torch.relu(self.self_proj(decomp[:, start:end]))
            scores = torch.bmm(block, block.transpose(1, 2))
            attended = torch.bmm(torch.softmax(scores, dim=-1), context[:, start:end])
            local[:, start:end] = attended
        mixed = torch.cat([local, aligned, context, aligned - context], dim=-1)
        out, _ = self.out(mixed)
        return out


class CurriculumPointerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        unk_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        hidden: int = 128,
        decoder_hidden: int = 256,
        block_size: int = 200,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.ial = IntrospectiveAlignmentLayer(hidden, block_size)
        self.init_h = nn.Linear(hidden, decoder_hidden)
        self.init_c = nn.Linear(hidden, decoder_hidden)
        self.decoder = nn.LSTMCell(hidden + hidden, decoder_hidden)
        self.att_y = nn.Linear(hidden, decoder_hidden, bias=False)
        self.att_h = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.att_q = nn.Linear(hidden, decoder_hidden, bias=False)
        self.att_v = nn.Linear(decoder_hidden, 1, bias=False)
        self.vocab_proj = nn.Linear(decoder_hidden, vocab_size)
        self.ptr_switch = nn.Linear(decoder_hidden + decoder_hidden + hidden, 1)

    def encode(self, context_ids: Any, question_ids: Any) -> tuple[Any, Any, Any, Any]:
        context_mask = context_ids != self.pad_token_id
        question_mask = question_ids != self.pad_token_id
        c, _ = self.encoder(self.embedding(context_ids))
        q, _ = self.encoder(self.embedding(question_ids))
        y = self.ial(c, q, question_mask)
        q_vec = q.masked_fill(~question_mask.unsqueeze(-1), 0.0).sum(dim=1) / question_mask.sum(dim=1, keepdim=True).clamp(min=1)
        y_vec = y.masked_fill(~context_mask.unsqueeze(-1), 0.0).sum(dim=1) / context_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return y, q_vec, context_mask, y_vec

    def _step(self, y: Any, q_vec: Any, context_ids: Any, context_mask: Any, prev_emb: Any, h: Any, c: Any) -> tuple[Any, Any, Any, Any]:
        scores = self.att_v(torch.tanh(self.att_y(y) + self.att_h(h).unsqueeze(1) + self.att_q(q_vec).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~context_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        y_t = torch.bmm(attn.unsqueeze(1), y).squeeze(1)
        h, c = self.decoder(torch.cat([y_t, prev_emb], dim=-1), (h, c))
        vocab_dist = torch.softmax(self.vocab_proj(h), dim=-1)
        p_gen = torch.sigmoid(self.ptr_switch(torch.cat([c, h, y_t], dim=-1)))
        copy_dist = torch.zeros(context_ids.size(0), self.vocab_size, dtype=vocab_dist.dtype, device=vocab_dist.device)
        copy_dist.scatter_add_(1, context_ids.clamp(max=self.vocab_size - 1), attn)
        final = p_gen * vocab_dist + (1.0 - p_gen) * copy_dist
        return final, attn, h, c

    def forward(self, context_ids: Any, question_ids: Any, answer_ids: Any | None = None, max_answer_len: int | None = None) -> CpgOutput:
        y, q_vec, context_mask, y_vec = self.encode(context_ids, question_ids)
        h = torch.tanh(self.init_h(y_vec))
        c = torch.tanh(self.init_c(y_vec))
        steps = answer_ids.size(1) - 1 if answer_ids is not None else max_answer_len or 32
        prev = torch.full((context_ids.size(0),), self.bos_token_id, dtype=torch.long, device=context_ids.device)
        logits = []
        attentions = []
        losses = []
        for t in range(steps):
            final, attn, h, c = self._step(y, q_vec, context_ids, context_mask, self.embedding(prev), h, c)
            logits.append(torch.log(final.clamp(min=1e-9)))
            attentions.append(attn)
            if answer_ids is not None:
                target = answer_ids[:, t + 1]
                losses.append(torch.nn.functional.nll_loss(torch.log(final.clamp(min=1e-9)), target, ignore_index=self.pad_token_id))
                prev = target
            else:
                prev = final.argmax(dim=-1)
        loss = sum(losses) / max(1, len(losses)) if losses else None
        return CpgOutput(loss, torch.stack(logits, dim=1), torch.stack(attentions, dim=1))
