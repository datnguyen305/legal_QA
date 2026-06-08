"""RE3QA unified retrieve-read-rerank model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Re3Output:
    loss: Any
    retrieve_logits: Any
    start_logits: Any
    end_logits: Any
    rerank_logits: Any


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
        raise SystemExit("RE3QA requires: python3 -m pip install -r requirements-models.txt")


class Re3QA(nn.Module):
    def __init__(self, base_model: str, early_layer: int = 3, max_candidates: int = 5) -> None:
        _missing_deps()
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model, output_hidden_states=True)
        hidden = self.encoder.config.hidden_size
        self.early_layer = early_layer
        self.max_candidates = max_candidates
        self.retrieve_att = nn.Linear(hidden, 1)
        self.retrieve_head = nn.Linear(hidden, 2)
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)
        self.rerank_att = nn.Linear(hidden, 1)
        self.rerank_head = nn.Linear(hidden, 1)

    def _summarize(self, hidden: Any, mask: Any) -> Any:
        scores = self.retrieve_att(hidden).squeeze(-1).masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)

    def _candidate_spans(self, start_logits: Any, end_logits: Any, sequence_ids: Any | None = None) -> list[list[tuple[int, int, float]]]:
        spans = []
        for b in range(start_logits.size(0)):
            candidates = []
            starts = torch.topk(start_logits[b], k=min(10, start_logits.size(1))).indices.tolist()
            ends = torch.topk(end_logits[b], k=min(10, end_logits.size(1))).indices.tolist()
            for s in starts:
                for e in ends:
                    if e >= s and e - s <= 30:
                        candidates.append((s, e, float(start_logits[b, s] + end_logits[b, e])))
            candidates.sort(key=lambda x: x[2], reverse=True)
            kept = []
            for s, e, score in candidates:
                if all(e < ks or s > ke for ks, ke, _ in kept):
                    kept.append((s, e, score))
                if len(kept) >= self.max_candidates:
                    break
            spans.append(kept or [(0, 0, 0.0)])
        return spans

    def _rerank(self, final_hidden: Any, candidates: list[list[tuple[int, int, float]]]) -> Any:
        rows = []
        for b, spans in enumerate(candidates):
            scores = []
            for s, e, _ in spans:
                span_hidden = final_hidden[b, s : e + 1]
                attn = torch.softmax(self.rerank_att(span_hidden).squeeze(-1), dim=0)
                vec = (attn.unsqueeze(-1) * span_hidden).sum(dim=0)
                scores.append(self.rerank_head(torch.tanh(vec)).squeeze(-1))
            while len(scores) < self.max_candidates:
                scores.append(torch.tensor(-1e4, device=final_hidden.device, dtype=final_hidden.dtype))
            rows.append(torch.stack(scores[: self.max_candidates]))
        return torch.stack(rows)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
        retrieve_labels: Any | None = None,
        start_positions: Any | None = None,
        end_positions: Any | None = None,
        rerank_labels: Any | None = None,
    ) -> Re3Output:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = enc.hidden_states
        early = hidden_states[min(self.early_layer, len(hidden_states) - 1)]
        final = enc.last_hidden_state
        retrieve_logits = self.retrieve_head(torch.tanh(self._summarize(early, attention_mask)))
        start_logits = self.start_head(final).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        end_logits = self.end_head(final).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        candidates = self._candidate_spans(start_logits.detach(), end_logits.detach())
        rerank_logits = self._rerank(final, candidates)
        losses = []
        if retrieve_labels is not None:
            losses.append(torch.nn.functional.cross_entropy(retrieve_logits, retrieve_labels))
        if start_positions is not None and end_positions is not None:
            losses.append(torch.nn.functional.cross_entropy(start_logits, start_positions))
            losses.append(torch.nn.functional.cross_entropy(end_logits, end_positions))
        if rerank_labels is not None:
            losses.append(torch.nn.functional.cross_entropy(rerank_logits, rerank_labels))
        return Re3Output(sum(losses) if losses else None, retrieve_logits, start_logits, end_logits, rerank_logits)
