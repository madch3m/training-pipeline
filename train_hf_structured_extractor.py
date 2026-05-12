from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def split_messages_for_sft(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each example must contain at least two messages.")

    completion = messages[-1]
    prompt = messages[:-1]
    if completion.get("role") != "assistant":
        raise ValueError("The final message must be an assistant message.")

    return {
        "prompt": prompt,
        "completion": [completion],
    }


def dataset_needs_conversion(column_names: list[str]) -> bool:
    return "messages" in column_names and not {"prompt", "completion"}.issubset(set(column_names))


def count_jsonl_rows(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return sum(1 for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_mlflow_params(args: argparse.Namespace, train_count: int, valid_count: int) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "train_jsonl": args.train_jsonl,
        "valid_jsonl": args.valid_jsonl,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
        "packing": args.packing,
        "use_lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_examples": train_count,
        "validation_examples": valid_count,
    }


def write_run_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    summary_path = Path(output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face model for structured syllabus extraction."
    )
    parser.add_argument("--train-jsonl", default="data/finetune/train.jsonl")
    parser.add_argument("--valid-jsonl", default="data/finetune/valid.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default="artifacts/hf_syllabus_extractor")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mlflow-experiment", default="syllabus-structured-extraction")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    import syllabus_torch_compat

    syllabus_torch_compat.apply_torch_ao_compat_patches()

    from datasets import load_dataset
    import mlflow
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if not args.disable_mlflow:
        mlflow.set_experiment(args.mlflow_experiment)

    dataset_dict = load_dataset(
        "json",
        data_files={
            "train": args.train_jsonl,
            "validation": args.valid_jsonl,
        },
    )

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"] if len(dataset_dict["validation"]) > 0 else None

    if dataset_needs_conversion(train_dataset.column_names):
        train_dataset = train_dataset.map(split_messages_for_sft)
    if eval_dataset is not None and dataset_needs_conversion(eval_dataset.column_names):
        eval_dataset = eval_dataset.map(split_messages_for_sft)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count = count_jsonl_rows(args.train_jsonl)
    valid_count = count_jsonl_rows(args.valid_jsonl)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=args.packing,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none" if args.disable_mlflow else "mlflow",
        seed=args.seed,
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    run_context = mlflow.start_run(run_name=args.mlflow_run_name) if not args.disable_mlflow else nullcontext()
    with run_context:
        if not args.disable_mlflow:
            mlflow.log_params(build_mlflow_params(args, train_count, valid_count))
            mlflow.set_tags(
                {
                    "pipeline": "syllabus_structured_extraction",
                    "trainer": "trl.SFTTrainer",
                    "task": "structured_json_generation",
                }
            )

        trainer = SFTTrainer(
            model=args.model_name,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        summary_path = write_run_summary(
            output_dir,
            {
                "train_metrics": train_result.metrics,
                "train_examples": train_count,
                "validation_examples": valid_count,
                "output_dir": str(output_dir),
            },
        )

        if not args.disable_mlflow:
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in train_result.metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="summaries")
            mlflow.log_artifacts(str(output_dir), artifact_path="model_artifacts")


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
