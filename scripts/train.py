import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from inspect import signature
from pathlib import Path

import numpy as np
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from question_encoder.config import DataConfig, ModelConfig, TrainConfig
from question_encoder.data import (
    compute_class_weights,
    compute_class_weights_multi,
    encode_labels,
    encode_labels_multi,
    load_dataframe,
    stratified_split,
    to_dataset,
    tokenize_dataset,
)
from question_encoder.metrics import build_compute_metrics, build_compute_metrics_multi
from question_encoder.modeling import load_model, load_tokenizer
from question_encoder.trainer import WeightedTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ModernBERT-base classifier")

    parser.add_argument("--data_path", default="data/data.csv")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="category")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--multi_label",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--label_sep", default="|")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--output_dir", default="outputs/modernbert-base")
    parser.add_argument("--logging_dir", default="runs/modernbert-base")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--use_class_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    return parser


def _maybe_sample(df, max_samples: int | None, seed: int):
    if max_samples is None or max_samples >= len(df):
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def _tokenize_split(df, tokenizer, data_cfg, model_cfg):
    if len(df) == 0:
        return None
    return tokenize_dataset(
        to_dataset(df),
        tokenizer=tokenizer,
        text_col=data_cfg.text_col,
        label_col=data_cfg.label_col,
        max_length=model_cfg.max_length,
    )


def _resolve_encoding(encoding, multi_label: bool):
    """Return (num_labels, label2id, id2label, problem_type) from either encoding type."""
    if multi_label:
        return encoding.num_labels, encoding.class2idx, encoding.idx2class, "multi_label_classification"
    return len(encoding.label2id), encoding.label2id, encoding.id2label, None


def _build_class_weights(train_df, num_labels, data_cfg, train_cfg):
    if not train_cfg.use_class_weights:
        return None
    if data_cfg.multi_label:
        labels_matrix = np.array(train_df["labels"].tolist())
        return compute_class_weights_multi(labels_matrix, num_labels=num_labels)
    return compute_class_weights(train_df["labels"].to_numpy(), num_labels=num_labels)


def _save_artifacts(
    output_dir,
    data_cfg,
    model_cfg,
    train_cfg,
    resolved_output_dir,
    resolved_logging_dir,
    label2id,
    id2label,
    split_sizes,
    logging_steps,
    warmup_steps,
    steps_per_epoch,
    effective_batch_size,
    train_metrics,
    val_metrics,
    test_metrics,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "label_mapping.json").open("w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    with (output_dir / "split_sizes.json").open("w") as f:
        json.dump(split_sizes, f, indent=2)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data": asdict(data_cfg),
            "model": asdict(model_cfg),
            "train": {
                **asdict(train_cfg),
                "resolved_output_dir": str(resolved_output_dir),
                "resolved_logging_dir": str(resolved_logging_dir),
                "computed_logging_steps": logging_steps,
                "computed_warmup_steps": warmup_steps,
                "steps_per_epoch": steps_per_epoch,
                "effective_batch_size": effective_batch_size,
            },
        },
        "split_sizes": split_sizes,
        "label_mapping": {"label2id": label2id, "id2label": id2label},
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = build_arg_parser().parse_args()

    data_cfg = DataConfig.from_namespace(args)
    model_cfg = ModelConfig.from_namespace(args)
    train_cfg = TrainConfig.from_namespace(args)

    resolved_output_dir = _next_available_dir(Path(train_cfg.output_dir))
    resolved_logging_dir = _next_available_dir(Path(train_cfg.logging_dir))
    if resolved_output_dir != Path(train_cfg.output_dir):
        print(f"output_dir exists, using: {resolved_output_dir}")
    if resolved_logging_dir != Path(train_cfg.logging_dir):
        print(f"logging_dir exists, using: {resolved_logging_dir}")

    set_seed(data_cfg.seed)

    # --- Data ---
    df = load_dataframe(data_cfg.data_path, data_cfg.text_col, data_cfg.label_col)
    if data_cfg.multi_label:
        df, encoding = encode_labels_multi(df, data_cfg.label_col, data_cfg.label_sep)
    else:
        df, encoding = encode_labels(df, data_cfg.label_col)

    train_df, val_df, test_df = stratified_split(
        df,
        label_col="labels",
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        test_ratio=data_cfg.test_ratio,
        seed=data_cfg.seed,
        multi_label=data_cfg.multi_label,
    )
    train_df = _maybe_sample(train_df, args.max_train_samples, data_cfg.seed)
    val_df = _maybe_sample(val_df, args.max_eval_samples, data_cfg.seed)
    test_df = _maybe_sample(test_df, args.max_test_samples, data_cfg.seed)

    # --- Tokenize ---
    tokenizer = load_tokenizer(model_cfg.model_name)
    train_ds = _tokenize_split(train_df, tokenizer, data_cfg, model_cfg)
    val_ds = _tokenize_split(val_df, tokenizer, data_cfg, model_cfg)
    test_ds = _tokenize_split(test_df, tokenizer, data_cfg, model_cfg)

    # --- Model ---
    num_labels, label2id, id2label, problem_type = _resolve_encoding(encoding, data_cfg.multi_label)
    model = load_model(
        model_cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type,
    )
    if train_cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    class_weights = _build_class_weights(train_df, num_labels, data_cfg, train_cfg)

    # --- Training args ---
    effective_batch_size = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch_size))
    logging_steps = train_cfg.logging_steps if train_cfg.logging_steps and train_cfg.logging_steps > 0 else max(1, steps_per_epoch // 10)
    warmup_steps = max(0, int(math.ceil(train_cfg.warmup_ratio * steps_per_epoch * float(train_cfg.num_train_epochs))))

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(resolved_logging_dir)

    load_best = val_ds is not None
    training_args = TrainingArguments(
        output_dir=str(resolved_output_dir),
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        max_grad_norm=train_cfg.max_grad_norm,
        logging_steps=logging_steps,
        logging_strategy="steps",
        save_total_limit=train_cfg.save_total_limit,
        eval_strategy="epoch" if load_best else "no",
        save_strategy="epoch",
        load_best_model_at_end=load_best,
        metric_for_best_model="macro_f1" if load_best else None,
        greater_is_better=True if load_best else None,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=["tensorboard"],
        seed=data_cfg.seed,
        data_seed=data_cfg.seed,
    )

    # Trainer
    if data_cfg.multi_label:
        compute_metrics = build_compute_metrics_multi(id2label, threshold=data_cfg.threshold)
    else:
        compute_metrics = build_compute_metrics(id2label)

    callbacks = []
    if val_ds is not None and train_cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.early_stopping_patience,
                early_stopping_threshold=train_cfg.early_stopping_threshold,
            )
        )

    trainer_kwargs = {}
    trainer_init_params = signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    # Train + evaluate
    train_result = trainer.train()
    trainer.remove_callback(EarlyStoppingCallback)
    trainer.save_model(str(resolved_output_dir))
    tokenizer.save_pretrained(str(resolved_output_dir))

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    val_metrics = {}
    if val_ds is not None:
        val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
        trainer.log_metrics("val", val_metrics)
        trainer.save_metrics("val", val_metrics)

    test_metrics = {}
    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    # Save
    split_sizes = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    _save_artifacts(
        resolved_output_dir,
        data_cfg,
        model_cfg,
        train_cfg,
        resolved_output_dir,
        resolved_logging_dir,
        label2id,
        id2label,
        split_sizes,
        logging_steps,
        warmup_steps,
        steps_per_epoch,
        effective_batch_size,
        train_metrics,
        val_metrics,
        test_metrics,
    )


if __name__ == "__main__":
    main()
