"""Run the syllabus URL → text → regex → SFT dataset → HF training pipeline from Python (e.g. Jupyter).

Example:

    from pathlib import Path
    from pipeline_runner import SyllabusPipelineConfig, find_repo_root, run_syllabus_training_pipeline

    cfg = SyllabusPipelineConfig.for_repo(find_repo_root())
    cfg.ingest_max_rows = 5  # smoke test
    run_syllabus_training_pipeline(cfg, steps=("csv", "ingest", "process", "build"))  # skip train
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from build_finetune_dataset import build_finetune_dataset
from ingest_jsonl_from_urls import ingest_jsonl_from_urls
from process_syllabi_jsonl import process_jsonl
from syllabi_index_csv_to_jsonl import csv_to_jsonl
from train_hf_structured_extractor import parse_args as train_parse_args, train_model as train_hf_model
from validate_training_readiness import ReadinessThresholds, evaluate_readiness


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: cwd) until ``build_finetune_dataset.py`` exists."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "build_finetune_dataset.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find training_pipeline repo root (no build_finetune_dataset.py in cwd or parents). "
        "Open the notebook from the repo or set repo_root manually."
    )


@dataclass
class SyllabusPipelineConfig:
    """Filesystem layout and hyperparameters for the full training pipeline."""

    repo_root: Path
    index_csv: Path
    url_jsonl: Path
    text_jsonl: Path
    entities_jsonl: Path
    train_jsonl: Path
    valid_jsonl: Path
    model_output_dir: Path
    ingest_sleep_seconds: float = 1.0
    ingest_timeout: float = 25.0
    ingest_max_rows: int | None = None
    csv_max_rows: int | None = None
    validation_ratio: float = 0.1
    max_text_chars: int = 12_000
    finetune_seed: int = 13
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    train_seed: int = 13
    bf16: bool = True
    fp16: bool = False
    disable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "syllabus-structured-extraction"
    mlflow_run_name: str | None = None
    allow_empty_finetune: bool = False
    enforce_training_gates: bool = True
    readiness_output_json: Path | None = None
    min_parse_rate: float = 0.995
    min_schema_clean_rate: float = 0.995
    max_document_id_overlap: int = 0
    max_domain_overlap_ratio: float = 0.70
    # SyllabusFacts teacher-distillation pipeline (used by stages: label_deepseek,
    # label_gpt4o_golden, label_gpt4o_arbitrate, merge_labels, build_facts).
    deepseek_facts_jsonl: Path = Path("data/labeled/deepseek_facts.jsonl")
    gpt4o_golden_jsonl: Path = Path("data/labeled/gpt4o_golden.jsonl")
    gpt4o_arbitration_jsonl: Path = Path("data/labeled/gpt4o_arbitration.jsonl")
    teacher_labels_jsonl: Path = Path("data/labeled/teacher_labels.jsonl")
    golden_facts_jsonl: Path = Path("data/labeled/golden_facts.jsonl")
    facts_train_jsonl: Path = Path("data/finetune_facts/train.jsonl")
    facts_valid_jsonl: Path = Path("data/finetune_facts/valid.jsonl")
    deepseek_model: str = "deepseek-chat"
    gpt4o_model: str = "gpt-4o-2024-11-20"
    label_max_text_chars: int = 20_000
    golden_ids_file: Path | None = None
    arbitration_ids_file: Path | None = None

    @classmethod
    def for_repo(cls, repo_root: Path) -> SyllabusPipelineConfig:
        rr = repo_root.resolve()
        return cls(
            repo_root=rr,
            index_csv=rr / "us_freshman_core_syllabi_index.csv",
            url_jsonl=rr / "data" / "ingested" / "us_freshman_core_syllabi_urls.jsonl",
            text_jsonl=rr / "data" / "ingested" / "us_freshman_core_syllabi_with_text.jsonl",
            entities_jsonl=rr / "data" / "labeled" / "syllabus_entities.jsonl",
            train_jsonl=rr / "data" / "finetune" / "train.jsonl",
            valid_jsonl=rr / "data" / "finetune" / "valid.jsonl",
            model_output_dir=rr / "artifacts" / "hf_syllabus_extractor",
            mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or (rr / "mlruns").resolve().as_uri(),
            deepseek_facts_jsonl=rr / "data" / "labeled" / "deepseek_facts.jsonl",
            gpt4o_golden_jsonl=rr / "data" / "labeled" / "gpt4o_golden.jsonl",
            gpt4o_arbitration_jsonl=rr / "data" / "labeled" / "gpt4o_arbitration.jsonl",
            teacher_labels_jsonl=rr / "data" / "labeled" / "teacher_labels.jsonl",
            golden_facts_jsonl=rr / "data" / "labeled" / "golden_facts.jsonl",
            facts_train_jsonl=rr / "data" / "finetune_facts" / "train.jsonl",
            facts_valid_jsonl=rr / "data" / "finetune_facts" / "valid.jsonl",
        )


def _chdir_repo(cfg: SyllabusPipelineConfig) -> None:
    os.chdir(cfg.repo_root)


def step_csv_to_jsonl(cfg: SyllabusPipelineConfig) -> int:
    _chdir_repo(cfg)
    return csv_to_jsonl(
        cfg.index_csv,
        cfg.url_jsonl,
        max_rows=cfg.csv_max_rows,
        skip_empty_url=True,
    )


def step_ingest(cfg: SyllabusPipelineConfig) -> tuple[int, int, int]:
    _chdir_repo(cfg)
    return ingest_jsonl_from_urls(
        cfg.url_jsonl,
        cfg.text_jsonl,
        sleep_seconds=cfg.ingest_sleep_seconds,
        timeout=cfg.ingest_timeout,
        max_rows=cfg.ingest_max_rows,
    )


def step_process(cfg: SyllabusPipelineConfig) -> list[dict]:
    _chdir_repo(cfg)
    return process_jsonl(cfg.text_jsonl, cfg.entities_jsonl)


def step_build_finetune(cfg: SyllabusPipelineConfig) -> tuple[list[dict], list[dict]]:
    _chdir_repo(cfg)
    return build_finetune_dataset(
        cfg.text_jsonl,
        cfg.train_jsonl,
        cfg.valid_jsonl,
        validation_ratio=cfg.validation_ratio,
        max_text_chars=cfg.max_text_chars,
        seed=cfg.finetune_seed,
        allow_empty_outputs=cfg.allow_empty_finetune,
    )


def build_train_argv(cfg: SyllabusPipelineConfig) -> list[str]:
    argv = [
        "train_hf_structured_extractor.py",
        "--train-jsonl",
        str(cfg.train_jsonl),
        "--valid-jsonl",
        str(cfg.valid_jsonl),
        "--model-name",
        cfg.model_name,
        "--output-dir",
        str(cfg.model_output_dir),
        "--num-train-epochs",
        str(cfg.num_train_epochs),
        "--learning-rate",
        str(cfg.learning_rate),
        "--per-device-train-batch-size",
        str(cfg.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(cfg.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(cfg.gradient_accumulation_steps),
        "--max-length",
        str(cfg.max_length),
        "--seed",
        str(cfg.train_seed),
        "--mlflow-experiment",
        cfg.mlflow_experiment,
    ]
    if cfg.mlflow_tracking_uri:
        argv.extend(["--mlflow-tracking-uri", cfg.mlflow_tracking_uri])
    if cfg.mlflow_run_name:
        argv.extend(["--mlflow-run-name", cfg.mlflow_run_name])
    if cfg.bf16:
        argv.append("--bf16")
    if cfg.fp16:
        argv.append("--fp16")
    if cfg.disable_mlflow:
        argv.append("--disable-mlflow")
    return argv


def step_train(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    if cfg.enforce_training_gates:
        thresholds = ReadinessThresholds(
            min_parse_rate=cfg.min_parse_rate,
            min_schema_clean_rate=cfg.min_schema_clean_rate,
            max_document_id_overlap=cfg.max_document_id_overlap,
            max_domain_overlap_ratio=cfg.max_domain_overlap_ratio,
        )
        report = evaluate_readiness(cfg.train_jsonl, cfg.valid_jsonl, thresholds)
        if cfg.readiness_output_json:
            cfg.readiness_output_json.parent.mkdir(parents=True, exist_ok=True)
            cfg.readiness_output_json.write_text(
                json.dumps(report, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        if not report["ok"]:
            raise ValueError(
                "Training readiness gates failed; inspect report violations before launching training."
            )
    argv = build_train_argv(cfg)
    old = sys.argv
    sys.argv = argv
    try:
        train_hf_model(train_parse_args())
    finally:
        sys.argv = old


def step_label_deepseek(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    cmd = [
        sys.executable, "label_with_deepseek.py",
        "--input-jsonl", str(cfg.text_jsonl),
        "--output-jsonl", str(cfg.deepseek_facts_jsonl),
        "--model", cfg.deepseek_model,
        "--max-text-chars", str(cfg.label_max_text_chars),
    ]
    subprocess.run(cmd, check=True)


def step_label_gpt4o_golden(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    if cfg.golden_ids_file is None:
        raise ValueError("cfg.golden_ids_file must be set for label_gpt4o_golden")
    cmd = [
        sys.executable, "label_with_gpt4o.py",
        "--input-jsonl", str(cfg.text_jsonl),
        "--output-jsonl", str(cfg.gpt4o_golden_jsonl),
        "--model", cfg.gpt4o_model,
        "--mode", "golden",
        "--golden-ids", str(cfg.golden_ids_file),
        "--max-text-chars", str(cfg.label_max_text_chars),
    ]
    subprocess.run(cmd, check=True)


def step_label_gpt4o_arbitrate(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    if cfg.arbitration_ids_file is None:
        raise ValueError("cfg.arbitration_ids_file must be set for label_gpt4o_arbitrate")
    cmd = [
        sys.executable, "label_with_gpt4o.py",
        "--input-jsonl", str(cfg.text_jsonl),
        "--output-jsonl", str(cfg.gpt4o_arbitration_jsonl),
        "--model", cfg.gpt4o_model,
        "--mode", "arbitrate",
        "--arbitration-ids", str(cfg.arbitration_ids_file),
        "--max-text-chars", str(cfg.label_max_text_chars),
    ]
    subprocess.run(cmd, check=True)


def step_merge_labels(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    cmd = [
        sys.executable, "merge_teacher_labels.py",
        "--deepseek-jsonl", str(cfg.deepseek_facts_jsonl),
        "--gpt4o-jsonl", str(cfg.gpt4o_arbitration_jsonl),
        "--golden-jsonl", str(cfg.golden_facts_jsonl),
        "--output-jsonl", str(cfg.teacher_labels_jsonl),
    ]
    subprocess.run(cmd, check=True)


def step_build_facts(cfg: SyllabusPipelineConfig) -> None:
    _chdir_repo(cfg)
    cmd = [
        sys.executable, "build_facts_dataset.py",
        "--text-jsonl", str(cfg.text_jsonl),
        "--labels-jsonl", str(cfg.teacher_labels_jsonl),
        "--train-output", str(cfg.facts_train_jsonl),
        "--valid-output", str(cfg.facts_valid_jsonl),
        "--max-text-chars", str(cfg.max_text_chars),
        "--validation-ratio", str(cfg.validation_ratio),
        "--seed", str(cfg.finetune_seed),
    ]
    subprocess.run(cmd, check=True)


def run_syllabus_training_pipeline(
    cfg: SyllabusPipelineConfig,
    *,
    steps: Sequence[str] = ("csv", "ingest", "process", "build", "train"),
) -> None:
    """Run pipeline stages in order.

    Legacy stages: ``csv``, ``ingest``, ``process``, ``build``, ``train``.
    Teacher-distillation stages (SyllabusFacts target):
      ``label_deepseek``, ``label_gpt4o_golden``, ``label_gpt4o_arbitrate``,
      ``merge_labels``, ``build_facts``.
    """
    allowed = {
        "csv", "ingest", "process", "build", "train",
        "label_deepseek", "label_gpt4o_golden", "label_gpt4o_arbitrate",
        "merge_labels", "build_facts",
    }
    for name in steps:
        if name not in allowed:
            raise ValueError(f"Unknown step {name!r}; allowed: {sorted(allowed)}")

    for name in steps:
        if name == "csv":
            n = step_csv_to_jsonl(cfg)
            print(f"[csv] Wrote {n} URL JSONL records → {cfg.url_jsonl}")
        elif name == "ingest":
            ok, skipped, failed = step_ingest(cfg)
            print(f"[ingest] ok={ok} skipped={skipped} failed={failed} → {cfg.text_jsonl}")
        elif name == "process":
            rows = step_process(cfg)
            print(f"[process] {len(rows)} labeled rows → {cfg.entities_jsonl}")
        elif name == "build":
            train, valid = step_build_finetune(cfg)
            print(f"[build] train={len(train)} valid={len(valid)} → {cfg.train_jsonl}")
        elif name == "train":
            print(f"[train] Starting LoRA SFT → {cfg.model_output_dir}")
            step_train(cfg)
            print("[train] Done.")
        elif name == "label_deepseek":
            print(f"[label_deepseek] → {cfg.deepseek_facts_jsonl}")
            step_label_deepseek(cfg)
        elif name == "label_gpt4o_golden":
            print(f"[label_gpt4o_golden] → {cfg.gpt4o_golden_jsonl}")
            step_label_gpt4o_golden(cfg)
        elif name == "label_gpt4o_arbitrate":
            print(f"[label_gpt4o_arbitrate] → {cfg.gpt4o_arbitration_jsonl}")
            step_label_gpt4o_arbitrate(cfg)
        elif name == "merge_labels":
            print(f"[merge_labels] → {cfg.teacher_labels_jsonl}")
            step_merge_labels(cfg)
        elif name == "build_facts":
            print(f"[build_facts] → {cfg.facts_train_jsonl}")
            step_build_facts(cfg)
