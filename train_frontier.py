"""Train and compare multiple base models for study-plan extraction."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
)


@dataclass(frozen=True)
class FrontierCandidate:
    model_name: str
    slug: str
    output_dir: Path
    train_seconds: float
    eval_summary: dict[str, Any]


def slugify_model_name(name: str) -> str:
    return (
        name.lower()
        .replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace(" ", "-")
    )


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare two or more LoRA model candidates.")
    p.add_argument("--train-jsonl", default="data/finetune/train.jsonl")
    p.add_argument("--valid-jsonl", default="data/finetune/valid.jsonl")
    p.add_argument("--output-root", default="artifacts/frontier")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--eval-max-samples", type=int, default=100)
    p.add_argument("--disable-mlflow", action="store_true")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--report-json", default="artifacts/frontier/frontier_report.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def train_candidate(args: argparse.Namespace, model_name: str, output_root: Path) -> FrontierCandidate:
    slug = slugify_model_name(model_name)
    out_dir = output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python",
        "train_hf_structured_extractor.py",
        "--train-jsonl",
        args.train_jsonl,
        "--valid-jsonl",
        args.valid_jsonl,
        "--model-name",
        model_name,
        "--output-dir",
        str(out_dir),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
    ]
    if args.disable_mlflow:
        train_cmd.append("--disable-mlflow")

    start = time.perf_counter()
    if not args.dry_run:
        run_cmd(train_cmd)
    elapsed = time.perf_counter() - start

    eval_report = out_dir / "eval_report.json"
    eval_cmd = [
        "python",
        "evaluate_trained_extractor.py",
        "--adapter-path",
        str(out_dir),
        "--sft-jsonl",
        args.valid_jsonl,
        "--max-samples",
        str(args.eval_max_samples),
        "--report-json",
        str(eval_report),
        "--device-map",
        args.device_map,
    ]
    if not args.dry_run:
        run_cmd(eval_cmd)
        eval_data = json.loads(eval_report.read_text(encoding="utf-8"))
        eval_summary = eval_data.get("summary", {})
    else:
        eval_summary = {
            "mean_study_plan_macro_f1": 0.0,
            "mean_macro_f1_string_lists": 0.0,
            "json_parse_success_rate": 0.0,
        }

    return FrontierCandidate(
        model_name=model_name,
        slug=slug,
        output_dir=out_dir,
        train_seconds=elapsed,
        eval_summary=eval_summary,
    )


def candidate_score(c: FrontierCandidate) -> float:
    # Quality-first weighted score for balanced edge SLA.
    study_f1 = float(c.eval_summary.get("mean_study_plan_macro_f1", 0.0))
    struct_f1 = float(c.eval_summary.get("mean_macro_f1_string_lists", 0.0))
    parse_ok = float(c.eval_summary.get("json_parse_success_rate", 0.0))
    return (0.60 * study_f1) + (0.25 * struct_f1) + (0.15 * parse_ok)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates: list[FrontierCandidate] = []
    for model_name in args.models:
        print(f"[frontier] train+eval {model_name}")
        cand = train_candidate(args, model_name, output_root)
        candidates.append(cand)

    ranked = sorted(candidates, key=candidate_score, reverse=True)
    winner = ranked[0] if ranked else None
    report = {
        "candidates": [
            {
                "model_name": c.model_name,
                "slug": c.slug,
                "output_dir": str(c.output_dir),
                "train_seconds": c.train_seconds,
                "score": candidate_score(c),
                "eval_summary": c.eval_summary,
            }
            for c in ranked
        ],
        "winner": (
            {
                "model_name": winner.model_name,
                "slug": winner.slug,
                "output_dir": str(winner.output_dir),
                "score": candidate_score(winner),
            }
            if winner
            else None
        ),
    }

    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
