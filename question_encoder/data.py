from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class LabelEncoding:
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    encoder: LabelEncoder


@dataclass(frozen=True)
class MultiLabelEncoding:
    classes: List[str]
    class2idx: Dict[str, int]
    idx2class: Dict[int, str]
    num_labels: int


def load_dataframe(data_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    path_lower = data_path.lower()
    if path_lower.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif path_lower.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif path_lower.endswith(".json"):
        df = pd.read_json(data_path)
    elif path_lower.endswith(".jsonl"):
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use one of: .csv, .parquet, .json, .jsonl")
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def encode_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, LabelEncoding]:
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[label_col].astype(str))
    out = df.copy()
    out["labels"] = labels
    label2id = {label: int(i) for i, label in enumerate(encoder.classes_)}
    id2label = {int(i): label for label, i in label2id.items()}
    return out, LabelEncoding(label2id=label2id, id2label=id2label, encoder=encoder)


def encode_labels_multi(
    df: pd.DataFrame, label_col: str, label_sep: str = "|",
) -> Tuple[pd.DataFrame, MultiLabelEncoding]:
    parsed = df[label_col].astype(str).str.split(label_sep).apply(
        lambda parts: [p.strip() for p in parts if p.strip()]
    )
    all_labels = sorted({lbl for lbls in parsed for lbl in lbls})
    class2idx = {lbl: i for i, lbl in enumerate(all_labels)}
    idx2class = {i: lbl for lbl, i in class2idx.items()}
    num_labels = len(all_labels)

    def _to_binary(label_list: List[str]) -> List[float]:
        vec = [0.0] * num_labels
        for lbl in label_list:
            vec[class2idx[lbl]] = 1.0
        return vec

    out = df.copy()
    out["labels"] = parsed.apply(_to_binary)
    encoding = MultiLabelEncoding(
        classes=all_labels, class2idx=class2idx, idx2class=idx2class, num_labels=num_labels,
    )
    return out, encoding


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative")


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    multi_label: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if val_ratio == 0 and test_ratio == 0:
        return df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    stratify_col = None if multi_label else df[label_col]
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=stratify_col,
        random_state=seed,
    )

    if val_ratio == 0:
        return train_df, temp_df.iloc[0:0].copy(), temp_df
    if test_ratio == 0:
        return train_df, temp_df, temp_df.iloc[0:0].copy()

    stratify_temp = None if multi_label else temp_df[label_col]
    test_size = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size,
        stratify=stratify_temp,
        random_state=seed,
    )
    return train_df, val_df, test_df


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer, text_col: str, label_col: str, max_length: int) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    remove_columns = [text_col, label_col]
    return dataset.map(tokenize, batched=True, remove_columns=remove_columns)


def compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    classes = np.arange(num_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def compute_class_weights_multi(labels_matrix: np.ndarray, num_labels: int) -> torch.Tensor:
    """Balanced pos-weight per label for BCEWithLogitsLoss: neg_count / pos_count."""
    pos_counts = labels_matrix.sum(axis=0).astype(float)
    neg_counts = labels_matrix.shape[0] - pos_counts
    pos_counts = np.clip(pos_counts, 1.0, None)
    weights = neg_counts / pos_counts
    return torch.tensor(weights, dtype=torch.float)
