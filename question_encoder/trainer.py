from __future__ import annotations

from typing import Optional

import torch
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels.dtype == torch.float:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=w)
            loss = loss_fct(logits, labels)
        else:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
