## ModernBERT Classifier + Encoder

This project fine-tunes `answerdotai/ModernBERT-base` for single-label or multi-label text classification with Apple Silicon MPS support, TensorBoard logging, weighted loss for imbalance, and embedding export.

## Setup (uv)

```bash
uv sync
```

## Data format

- Input file types: `.csv`, `.parquet`, `.json`, `.jsonl`
- Default text column: `text`
- Default label column: `category`
- **Single-label** (default): one label per row (e.g. `sports`)
- **Multi-label** (`--multi_label`): pipe-separated labels per row (e.g. `sports|politics|tech`)

## Train

Recommended baseline on Mac (MPS):

```bash
uv run python scripts/train.py \
  --data_path data/data.csv \
  --text_col text \
  --label_col category \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --optim adamw_torch
```

Notes:
- Default split is `train/val` with `--test_ratio 0.0`.
- Ratios must sum to `1.0`.
- Early stopping is enabled when validation exists.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the script automatically.
- If `--output_dir` already exists, training auto-creates a new directory with a numeric suffix (for example `outputs/modernbert-base_1`, `outputs/modernbert-base_2`).

## Multi-label train

For datasets where each row can have multiple labels (pipe-separated in the label column):

```bash
uv run python scripts/train.py \
  --data_path data/data.csv \
  --text_col text \
  --label_col category \
  --multi_label \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --optim adamw_torch
```

Notes:
- Labels are pipe-separated by default (e.g. `sports|politics|tech`).
- Uses `BCEWithLogitsLoss` with per-label pos-weights for class imbalance.
- Metrics include hamming loss, subset accuracy (exact match), and per-label F1.
- Splits are random (not stratified) since sklearn stratification doesn't support multi-label.
- Inference auto-detects multi-label from the saved model config; use `--threshold` to tune the sigmoid cutoff.

## Training args (full reference)

| Arg | Default | Recommended | Notes |
|---|---|---|---|
| `--data_path` | `data/data.csv` | Set to your dataset path | Supports csv/parquet/json/jsonl |
| `--text_col` | `text` | Keep unless schema differs | Text feature column |
| `--label_col` | `category` | Keep unless schema differs | Label column |
| `--multi_label` | `False` | `True` for multi-select datasets | Enables multi-label mode |
| `--label_sep` | `\|` | Keep unless using different delimiter | Delimiter for multi-label strings |
| `--threshold` | `0.5` | Tune per task | Sigmoid cutoff for multi-label predictions |
| `--train_ratio` | `0.9` | `0.9` | Must sum with val/test to 1.0 |
| `--val_ratio` | `0.1` | `0.1` | Set `0` only if you want no eval |
| `--test_ratio` | `0.0` | `0.0` during tuning, `0.1` for final holdout | Optional test split |
| `--seed` | `42` | `42` | Reproducibility |
| `--model_name` | `answerdotai/ModernBERT-base` | Keep default | Base model |
| `--max_length` | `512` | `512` | High memory cost on MPS |
| `--output_dir` | `outputs/modernbert-base` | Keep or experiment-specific path | Model and metrics output |
| `--logging_dir` | `runs/modernbert-base` | Keep | TensorBoard logs |
| `--learning_rate` | `5e-5` | `5e-5` start, try `2e-5` if unstable | AdamW learning rate |
| `--weight_decay` | `0.0` | `0.01` if overfitting, else `0.0` | L2 regularization |
| `--num_train_epochs` | `3.0` | `3` start, up to `5` if underfitting | Early stopping can cut short |
| `--per_device_train_batch_size` | `8` | `8` on larger RAM Macs, `4` on tighter memory | Per-step train batch |
| `--per_device_eval_batch_size` | `8` | `8` or `16` if memory allows | Eval batch |
| `--gradient_accumulation_steps` | `8` | `8` | Effective batch = train batch * grad accum |
| `--warmup_ratio` | `0.05` | `0.05` | Converted internally to `warmup_steps` for Transformers v5+ |
| `--optim` | `adamw_torch` | `adamw_torch` | Default and recommended |
| `--lr_scheduler_type` | `linear` | `linear` | Stable default |
| `--max_grad_norm` | `1.0` | `1.0` | Gradient clipping |
| `--logging_steps` | `0` | `0` | `0` means auto (~10 logs/epoch) |
| `--early_stopping_patience` | `3` | `3` | Set `0` to disable |
| `--early_stopping_threshold` | `0.001` | `0.001` | Minimum eval improvement |
| `--gradient_checkpointing` | `False` | `True` if OOM, else `False` | Trades speed for memory |
| `--no-gradient_checkpointing` | N/A | Use to force off | Boolean inverse flag |
| `--save_total_limit` | `2` | `2` | Retained checkpoints |
| `--use_class_weights` | `True` | `True` on imbalanced labels | Weighted CE loss |
| `--no-use_class_weights` | N/A | Use when classes are balanced | Boolean inverse flag |
| `--max_train_samples` | `None` | Set for smoke tests | Limits train rows |
| `--max_eval_samples` | `None` | Set for smoke tests | Limits val rows |
| `--max_test_samples` | `None` | Set for smoke tests | Limits test rows |

## TensorBoard

```bash
uv run tensorboard --logdir runs
```

## Outputs

Training writes:
- `outputs/modernbert-base/results.json` (centralized config + split sizes + metrics)
- `outputs/modernbert-base/label_mapping.json`
- `outputs/modernbert-base/split_sizes.json`
- model/tokenizer artifacts under `outputs/modernbert-base/`

## Inference + Embeddings

Multi-label vs single-label is auto-detected from the saved model config. No extra flag needed.

Single text:

```bash
uv run python scripts/infer.py \
  --model_dir outputs/modernbert-base \
  --input_text "hello world"
```

Batch CSV:

```bash
uv run python scripts/infer.py \
  --model_dir outputs/modernbert-base \
  --input_csv data/data.csv \
  --text_col text \
  --output_csv predictions.csv
```

Multi-label models output `pred_labels` (pipe-separated) instead of `pred_label`. Use `--threshold` to adjust the sigmoid cutoff (default `0.5`).
