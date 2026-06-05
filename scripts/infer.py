import argparse
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from question_encoder.gpu import resolve_device
from question_encoder.infer import predict_csv, predict_texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference and embeddings")
    parser.add_argument("--model_dir", default="outputs/modernbert-base")
    parser.add_argument("--input_text", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--emit_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label_sep", default="|")
    return parser


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = build_arg_parser().parse_args()

    if not args.input_text and not args.input_csv:
        raise SystemExit("Provide --input_text or --input_csv")
    if args.input_text and args.input_csv:
        raise SystemExit("Use only one of --input_text or --input_csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = resolve_device()

    multi_label = getattr(model.config, "problem_type", None) == "multi_label_classification"

    if args.input_text:
        rows = predict_texts(
            model=model,
            tokenizer=tokenizer,
            texts=[args.input_text],
            max_length=args.max_length,
            batch_size=1,
            device=device,
            emit_embeddings=args.emit_embeddings,
            multi_label=multi_label,
            threshold=args.threshold,
            label_sep=args.label_sep,
        )
        if args.output_csv:
            import pandas as pd

            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        else:
            for row in rows:
                print(row)
        return

    predict_csv(
        model=model,
        tokenizer=tokenizer,
        input_csv=args.input_csv,
        text_col=args.text_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        emit_embeddings=args.emit_embeddings,
        output_csv=args.output_csv,
        multi_label=multi_label,
        threshold=args.threshold,
        label_sep=args.label_sep,
    )


if __name__ == "__main__":
    main()
