from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    problem_type: str | None = None,
):
    kwargs: dict = dict(num_labels=num_labels, id2label=id2label, label2id=label2id)
    if problem_type is not None:
        kwargs["problem_type"] = problem_type
    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_base_model(model):
    return model.base_model
