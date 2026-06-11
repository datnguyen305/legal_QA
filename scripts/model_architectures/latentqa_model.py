"""LatentQA-style stochastic selector network.

The decoder marginalizes each answer token over three latent sources: global
vocabulary, question copy, and context copy. This is a compact implementation of
the paper's discrete source-selector idea for this repository's tokenized Legal
QA data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LatentQAOutput:
    loss: Any
    logits: Any
    source_probs: Any


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
        raise SystemExit("LatentQA requires PyTorch. Install dependencies from requirements-models.txt.")


class LatentQA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        hidden: int = 128,
        decoder_hidden: int = 256,
    ) -> None:
        _missing_deps()
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.context_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.question_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(hidden * 2, decoder_hidden)
        self.init_c = nn.Linear(hidden * 2, decoder_hidden)
        self.decoder = nn.LSTMCell(hidden, decoder_hidden)
        self.ctx_att = nn.Linear(hidden, decoder_hidden, bias=False)
        self.q_att = nn.Linear(hidden, decoder_hidden, bias=False)
        self.dec_att = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.att_v = nn.Linear(decoder_hidden, 1, bias=False)
        self.vocab_proj = nn.Linear(decoder_hidden + hidden * 2, vocab_size)
        self.source_proj = nn.Linear(decoder_hidden + hidden * 2, 3)

    def _masked_mean(self, x: Any, mask: Any) -> Any:
        return x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _attend(self, memory: Any, mask: Any, h: Any, proj: Any) -> tuple[Any, Any]:
        scores = self.att_v(torch.tanh(proj(memory) + self.dec_att(h).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn.unsqueeze(1), memory).squeeze(1), attn

    def forward(
        self,
        context_ids: Any,
        question_ids: Any,
        answer_ids: Any | None = None,
        max_answer_len: int | None = None,
    ) -> LatentQAOutput:
        c_mask = context_ids != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        c_mem, _ = self.context_encoder(self.embedding(context_ids))
        q_mem, _ = self.question_encoder(self.embedding(question_ids))
        init = torch.cat([self._masked_mean(c_mem, c_mask), self._masked_mean(q_mem, q_mask)], dim=-1)
        h = torch.tanh(self.init_h(init))
        c = torch.tanh(self.init_c(init))
        steps = answer_ids.size(1) - 1 if answer_ids is not None else max_answer_len or 32
        prev = torch.full((context_ids.size(0),), self.bos_token_id, dtype=torch.long, device=context_ids.device)
        logits = []
        source_probs = []
        losses = []
        for t in range(steps):
            h, c = self.decoder(self.embedding(prev), (h, c))
            ctx_vec, ctx_attn = self._attend(c_mem, c_mask, h, self.ctx_att)
            q_vec, q_attn = self._attend(q_mem, q_mask, h, self.q_att)
            state = torch.cat([h, q_vec, ctx_vec], dim=-1)
            src = torch.softmax(self.source_proj(state), dim=-1)
            vocab_dist = torch.softmax(self.vocab_proj(state), dim=-1)
            q_copy = torch.zeros_like(vocab_dist)
            c_copy = torch.zeros_like(vocab_dist)
            q_copy.scatter_add_(1, question_ids.clamp(max=self.vocab_size - 1), q_attn)
            c_copy.scatter_add_(1, context_ids.clamp(max=self.vocab_size - 1), ctx_attn)
            final = src[:, 0:1] * vocab_dist + src[:, 1:2] * q_copy + src[:, 2:3] * c_copy
            logits.append(torch.log(final.clamp(min=1e-9)))
            source_probs.append(src)
            if answer_ids is not None:
                target = answer_ids[:, t + 1]
                losses.append(torch.nn.functional.nll_loss(torch.log(final.clamp(min=1e-9)), target, ignore_index=self.pad_token_id))
                prev = target
            else:
                prev = final.argmax(dim=-1)
        loss = sum(losses) / max(1, len(losses)) if losses else None
        return LatentQAOutput(loss, torch.stack(logits, dim=1), torch.stack(source_probs, dim=1))
