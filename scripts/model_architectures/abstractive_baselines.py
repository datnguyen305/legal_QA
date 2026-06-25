"""From-scratch abstractive QA baselines.

The models share a compact pointer-generator decoder and differ in the reader
state used by the decoder:

- DCMNPlusGenerator: dual co-matching features between context and question.
- MultiStyleGenerativeRC: style-conditioned encoder/decoder for answer style.
- GAQAGenerator: gated answer-aware question/context reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AbstractiveOutput:
    loss: Any
    logits: Any
    copy_attention: Any


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
        raise SystemExit("Abstractive baselines require PyTorch. Install dependencies from requirements-models.txt.")


class PointerDecoderMixin:
    def _masked_mean(self, x: Any, mask: Any) -> Any:
        return x.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def _decode(
        self,
        memory: Any,
        memory_mask: Any,
        source_ids: Any,
        init_state: Any,
        question_vec: Any,
        answer_ids: Any | None,
        max_answer_len: int | None,
    ) -> AbstractiveOutput:
        h = init_state
        prev = torch.full((source_ids.size(0),), self.bos_token_id, dtype=torch.long, device=source_ids.device)
        steps = answer_ids.size(1) - 1 if answer_ids is not None else max_answer_len or 64
        logits = []
        attentions = []
        losses = []
        for t in range(steps):
            emb = self.embedding(prev)
            scores = self.att_v(torch.tanh(self.att_mem(memory) + self.att_dec(h).unsqueeze(1))).squeeze(-1)
            scores = scores.masked_fill(~memory_mask, -1e4)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
            h = self.decoder(torch.cat([emb, ctx, question_vec], dim=-1), h)
            readout = torch.tanh(self.readout(torch.cat([h, ctx, question_vec, emb], dim=-1)))
            vocab_dist = torch.softmax(self.vocab_proj(readout), dim=-1)
            p_gen = torch.sigmoid(self.copy_gate(torch.cat([h, ctx, question_vec, emb], dim=-1)))
            copy_dist = torch.zeros(source_ids.size(0), self.vocab_size, dtype=vocab_dist.dtype, device=vocab_dist.device)
            copy_dist.scatter_add_(1, source_ids.clamp(max=self.vocab_size - 1), attn)
            final = p_gen * vocab_dist + (1.0 - p_gen) * copy_dist
            log_final = torch.log(final.clamp(min=1e-9))
            logits.append(log_final)
            attentions.append(attn)
            if answer_ids is not None:
                target = answer_ids[:, t + 1]
                losses.append(F.nll_loss(log_final, target, ignore_index=self.pad_token_id))
                prev = target
            else:
                prev = final.argmax(dim=-1)
        loss = sum(losses) / max(1, len(losses)) if losses else None
        return AbstractiveOutput(loss, torch.stack(logits, dim=1), torch.stack(attentions, dim=1))


class DCMNPlusGenerator(nn.Module, PointerDecoderMixin):
    """Dual co-matching reader with pointer generation."""

    def __init__(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, hidden: int = 128, decoder_hidden: int = 256) -> None:
        _missing_deps()
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.context_encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.question_encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.match_encoder = nn.GRU(hidden * 4, hidden // 2, batch_first=True, bidirectional=True)
        self.init = nn.Linear(hidden * 2, decoder_hidden)
        self.decoder = nn.GRUCell(hidden * 3, decoder_hidden)
        self.att_mem = nn.Linear(hidden, decoder_hidden, bias=False)
        self.att_dec = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.att_v = nn.Linear(decoder_hidden, 1, bias=False)
        self.readout = nn.Linear(decoder_hidden + hidden * 3, decoder_hidden)
        self.vocab_proj = nn.Linear(decoder_hidden, vocab_size)
        self.copy_gate = nn.Linear(decoder_hidden + hidden * 3, 1)

    def forward(self, context_ids: Any, question_ids: Any, answer_ids: Any | None = None, max_answer_len: int | None = None, style_ids: Any | None = None) -> AbstractiveOutput:
        c_mask = context_ids != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        c, _ = self.context_encoder(self.embedding(context_ids))
        q, _ = self.question_encoder(self.embedding(question_ids))
        sim = torch.bmm(c, q.transpose(1, 2)).masked_fill(~q_mask.unsqueeze(1), -1e4)
        c2q = torch.bmm(torch.softmax(sim, dim=-1), q)
        matched, _ = self.match_encoder(torch.cat([c, c2q, c - c2q, c * c2q], dim=-1))
        q_vec = self._masked_mean(q, q_mask)
        c_vec = self._masked_mean(matched, c_mask)
        init = torch.tanh(self.init(torch.cat([c_vec, q_vec], dim=-1)))
        return self._decode(matched, c_mask, context_ids, init, q_vec, answer_ids, max_answer_len)


class MultiStyleGenerativeRC(nn.Module, PointerDecoderMixin):
    """Masque-inspired multi-style generative reader.

    The repository has one gold answer style by default, but the architecture
    keeps a style embedding and style-conditioned decoder path so additional
    styles can be added without changing the model contract.
    """

    def __init__(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, hidden: int = 128, decoder_hidden: int = 256, num_styles: int = 4) -> None:
        _missing_deps()
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.style_embedding = nn.Embedding(num_styles, hidden)
        self.context_encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.question_encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.fuse = nn.Linear(hidden * 4, hidden)
        self.init = nn.Linear(hidden * 3, decoder_hidden)
        self.decoder = nn.GRUCell(hidden * 3, decoder_hidden)
        self.att_mem = nn.Linear(hidden, decoder_hidden, bias=False)
        self.att_dec = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.att_v = nn.Linear(decoder_hidden, 1, bias=False)
        self.readout = nn.Linear(decoder_hidden + hidden * 3, decoder_hidden)
        self.vocab_proj = nn.Linear(decoder_hidden, vocab_size)
        self.copy_gate = nn.Linear(decoder_hidden + hidden * 3, 1)

    def forward(self, context_ids: Any, question_ids: Any, answer_ids: Any | None = None, max_answer_len: int | None = None, style_ids: Any | None = None) -> AbstractiveOutput:
        c_mask = context_ids != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        if style_ids is None:
            style_ids = torch.zeros(context_ids.size(0), dtype=torch.long, device=context_ids.device)
        style_vec = self.style_embedding(style_ids)
        c, _ = self.context_encoder(self.embedding(context_ids) + style_vec.unsqueeze(1))
        q, _ = self.question_encoder(self.embedding(question_ids))
        q_vec = self._masked_mean(q, q_mask)
        q_exp = q_vec.unsqueeze(1).expand(-1, c.size(1), -1)
        memory = torch.tanh(self.fuse(torch.cat([c, q_exp, c * q_exp, style_vec.unsqueeze(1).expand_as(c)], dim=-1)))
        c_vec = self._masked_mean(memory, c_mask)
        init = torch.tanh(self.init(torch.cat([c_vec, q_vec, style_vec], dim=-1)))
        styled_q = q_vec + style_vec
        return self._decode(memory, c_mask, context_ids, init, styled_q, answer_ids, max_answer_len)


class GAQAGenerator(nn.Module, PointerDecoderMixin):
    """Gated attention QA generator."""

    def __init__(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int, hidden: int = 128, decoder_hidden: int = 256) -> None:
        _missing_deps()
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_token_id)
        self.context_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.question_encoder = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.gate = nn.Linear(hidden * 4, hidden)
        self.gated_encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.init = nn.Linear(hidden * 2, decoder_hidden)
        self.decoder = nn.GRUCell(hidden * 3, decoder_hidden)
        self.att_mem = nn.Linear(hidden, decoder_hidden, bias=False)
        self.att_dec = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.att_v = nn.Linear(decoder_hidden, 1, bias=False)
        self.readout = nn.Linear(decoder_hidden + hidden * 3, decoder_hidden)
        self.vocab_proj = nn.Linear(decoder_hidden, vocab_size)
        self.copy_gate = nn.Linear(decoder_hidden + hidden * 3, 1)

    def forward(self, context_ids: Any, question_ids: Any, answer_ids: Any | None = None, max_answer_len: int | None = None, style_ids: Any | None = None) -> AbstractiveOutput:
        c_mask = context_ids != self.pad_token_id
        q_mask = question_ids != self.pad_token_id
        c, _ = self.context_encoder(self.embedding(context_ids))
        q, _ = self.question_encoder(self.embedding(question_ids))
        sim = torch.bmm(c, q.transpose(1, 2)).masked_fill(~q_mask.unsqueeze(1), -1e4)
        q_att = torch.bmm(torch.softmax(sim, dim=-1), q)
        gate = torch.sigmoid(self.gate(torch.cat([c, q_att, c - q_att, c * q_att], dim=-1)))
        gated = gate * c + (1.0 - gate) * q_att
        memory, _ = self.gated_encoder(gated)
        q_vec = self._masked_mean(q, q_mask)
        c_vec = self._masked_mean(memory, c_mask)
        init = torch.tanh(self.init(torch.cat([c_vec, q_vec], dim=-1)))
        return self._decode(memory, c_mask, context_ids, init, q_vec, answer_ids, max_answer_len)
