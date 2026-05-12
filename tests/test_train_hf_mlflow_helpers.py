import argparse
import json
from pathlib import Path

from train_hf_structured_extractor import build_mlflow_params, count_jsonl_rows, write_run_summary


def test_count_jsonl_rows_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert count_jsonl_rows(path) == 2


def test_build_mlflow_params_includes_run_metadata():
    args = argparse.Namespace(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_jsonl="train.jsonl",
        valid_jsonl="valid.jsonl",
        num_train_epochs=3.0,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        weight_decay=0.0,
        warmup_ratio=0.03,
        max_length=4096,
        seed=13,
        packing=False,
        no_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    params = build_mlflow_params(args, train_count=100, valid_count=12)

    assert params["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert params["use_lora"] is True
    assert params["train_examples"] == 100
    assert params["validation_examples"] == 12


def test_write_run_summary_writes_json_file(tmp_path: Path):
    summary = {"train_metrics": {"loss": 1.23}, "train_examples": 5}
    path = write_run_summary(tmp_path, summary)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["train_metrics"]["loss"] == 1.23
