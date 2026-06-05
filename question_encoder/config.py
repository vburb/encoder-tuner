from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, fields


DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"


def _pick(ns: Namespace, cls: type) -> dict:
    """Extract only the keys that match dataclass fields from an argparse Namespace."""
    ns_vars = vars(ns)
    return {f.name: ns_vars[f.name] for f in fields(cls) if f.name in ns_vars}


@dataclass(frozen=True)
class DataConfig:
    data_path: str = "data/data.csv"
    text_col: str = "text"
    label_col: str = "category"
    multi_label: bool = False
    label_sep: str = "|"
    threshold: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42

    @classmethod
    def from_namespace(cls, ns: Namespace) -> DataConfig:
        return cls(**_pick(ns, cls))


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512

    @classmethod
    def from_namespace(cls, ns: Namespace) -> ModelConfig:
        return cls(**_pick(ns, cls))


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "outputs/modernbert-base"
    logging_dir: str = "runs/modernbert-base"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    logging_steps: int = 0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    gradient_checkpointing: bool = False
    save_total_limit: int = 2
    use_class_weights: bool = True

    @classmethod
    def from_namespace(cls, ns: Namespace) -> TrainConfig:
        return cls(**_pick(ns, cls))
