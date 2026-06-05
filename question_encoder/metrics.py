from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def build_compute_metrics(id2label: Dict[int, str]):
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "weighted_precision": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
            "macro_recall": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            "weighted_recall": recall_score(
                labels, preds, average="weighted", zero_division=0
            ),
        }

        report = classification_report(
            labels,
            preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        for label, stats in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            metrics[f"precision_{label}"] = stats.get("precision", 0.0)
            metrics[f"recall_{label}"] = stats.get("recall", 0.0)
            metrics[f"f1_{label}"] = stats.get("f1-score", 0.0)
        return metrics

    return compute_metrics


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_compute_metrics_multi(id2label: Dict[int, str], threshold: float = 0.5):
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = _sigmoid(logits)
        preds = (probs >= threshold).astype(int)
        labels_int = labels.astype(int)

        metrics = {
            "hamming_loss": hamming_loss(labels_int, preds),
            "subset_accuracy": accuracy_score(labels_int, preds),
            "macro_f1": f1_score(labels_int, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels_int, preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(
                labels_int, preds, average="macro", zero_division=0,
            ),
            "weighted_precision": precision_score(
                labels_int, preds, average="weighted", zero_division=0,
            ),
            "macro_recall": recall_score(
                labels_int, preds, average="macro", zero_division=0,
            ),
            "weighted_recall": recall_score(
                labels_int, preds, average="weighted", zero_division=0,
            ),
        }

        for i, name in enumerate(label_names):
            col_true = labels_int[:, i]
            col_pred = preds[:, i]
            metrics[f"precision_{name}"] = precision_score(
                col_true, col_pred, zero_division=0,
            )
            metrics[f"recall_{name}"] = recall_score(
                col_true, col_pred, zero_division=0,
            )
            metrics[f"f1_{name}"] = f1_score(col_true, col_pred, zero_division=0)

        return metrics

    return compute_metrics
