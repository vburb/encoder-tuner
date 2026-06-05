from __future__ import annotations

import json
from typing import Iterable, List, Optional

import pandas as pd
import torch

from .modeling import get_base_model, mean_pool


def _resolve_id2label(id2label):
    if not isinstance(id2label, dict):
        return {}
    out = {}
    for k, v in id2label.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def predict_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    emit_embeddings: bool,
    multi_label: bool = False,
    threshold: float = 0.5,
    label_sep: str = "|",
):
    model.eval()
    model.to(device)

    id2label = _resolve_id2label(model.config.id2label)
    encoder = get_base_model(model) if emit_embeddings else None
    rows = []

    with torch.no_grad():
        for batch_texts in _batched(texts, batch_size):
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits

            if multi_label:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            if emit_embeddings:
                enc_outputs = encoder(**encoded)
                embeddings = mean_pool(enc_outputs.last_hidden_state, encoded["attention_mask"])
            else:
                embeddings = None

            for i, text in enumerate(batch_texts):
                if multi_label:
                    mask = probs[i] >= threshold
                    pred_indices = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                    if isinstance(pred_indices, int):
                        pred_indices = [pred_indices]
                    pred_labels = [id2label.get(idx, str(idx)) for idx in pred_indices]
                    row = {
                        "text": text,
                        "pred_labels": label_sep.join(pred_labels),
                        "probs": json.dumps([float(x) for x in probs[i].cpu().tolist()]),
                    }
                else:
                    pred_id = int(torch.argmax(probs[i]).item())
                    pred_label = id2label.get(pred_id, str(pred_id))
                    row = {
                        "text": text,
                        "pred_id": pred_id,
                        "pred_label": pred_label,
                        "pred_prob": float(probs[i][pred_id].item()),
                        "probs": json.dumps([float(x) for x in probs[i].cpu().tolist()]),
                    }
                if emit_embeddings:
                    row["embedding"] = json.dumps(
                        [float(x) for x in embeddings[i].cpu().tolist()]
                    )
                rows.append(row)

    return rows


def predict_csv(
    model,
    tokenizer,
    input_csv: str,
    text_col: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    emit_embeddings: bool,
    output_csv: Optional[str],
    multi_label: bool = False,
    threshold: float = 0.5,
    label_sep: str = "|",
):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Missing column '{text_col}' in {input_csv}")
    texts = df[text_col].astype(str).tolist()
    rows = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        emit_embeddings=emit_embeddings,
        multi_label=multi_label,
        threshold=threshold,
        label_sep=label_sep,
    )
    out_df = pd.DataFrame(rows)
    if output_csv:
        out_df.to_csv(output_csv, index=False)
    return out_df
