"""Fault-tolerant parsing and metrics for structured JSON (SFT labels or model output)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Expected keys in assistant payloads produced by ``build_finetune_dataset``.
EXPECTED_ASSISTANT_KEYS = frozenset(
    {
        "document_id",
        "source_url",
        "course_codes",
        "instructors",
        "emails",
        "section_names",
        "assignments",
        "readings",
        "grading_weights",
        "due_dates",
        "course_dates",
        "concepts",
        "text_field",
        "entities",
    }
)


def parse_assistant_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse model or dataset JSON string; return ``(obj, None)`` or ``(None, error)``."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(data, dict):
        return None, "root_not_object"
    return data, None


def assistant_payload_from_sft_example(example: dict) -> tuple[str | None, str | None]:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "missing_messages"
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return None, "no_assistant_message"
    content = last.get("content")
    if not isinstance(content, str):
        return None, "assistant_content_not_string"
    return content, None


def validate_assistant_object(obj: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = EXPECTED_ASSISTANT_KEYS - set(obj.keys())
    if missing:
        issues.append(f"missing_keys:{sorted(missing)}")
    extra = set(obj.keys()) - EXPECTED_ASSISTANT_KEYS
    if extra:
        issues.append(f"extra_keys:{sorted(extra)}")
    for key in (
        "course_codes",
        "instructors",
        "emails",
        "section_names",
        "assignments",
        "readings",
        "grading_weights",
        "due_dates",
        "course_dates",
        "concepts",
        "entities",
    ):
        val = obj.get(key)
        if val is not None and not isinstance(val, list):
            issues.append(f"{key}_not_list")
    return issues


def evaluate_sft_jsonl(path: str | Path) -> dict[str, Any]:
    """Score teacher JSON in a train/valid JSONL (one chat example per line)."""
    file_path = Path(path)
    report: dict[str, Any] = {
        "path": str(file_path.resolve()),
        "lines": 0,
        "parseable_assistant_json": 0,
        "assistant_json_errors": 0,
        "schema_issues": 0,
        "error_samples": [],
    }
    if not file_path.exists():
        report["note"] = "file_missing"
        return report

    err_samples: list[dict[str, Any]] = []
    for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        report["lines"] += 1
        try:
            example = json.loads(stripped)
        except json.JSONDecodeError as exc:
            report["assistant_json_errors"] += 1
            if len(err_samples) < 5:
                err_samples.append({"line": line_no, "stage": "line_json", "error": str(exc)})
            continue

        raw, err = assistant_payload_from_sft_example(example)
        if err:
            report["assistant_json_errors"] += 1
            if len(err_samples) < 5:
                err_samples.append({"line": line_no, "stage": "messages", "error": err})
            continue
        assert raw is not None
        obj, perr = parse_assistant_json(raw)
        if perr:
            report["assistant_json_errors"] += 1
            if len(err_samples) < 5:
                err_samples.append({"line": line_no, "stage": "assistant_json", "error": perr})
            continue
        assert obj is not None
        issues = validate_assistant_object(obj)
        if issues:
            report["schema_issues"] += 1
            if len(err_samples) < 5:
                err_samples.append({"line": line_no, "stage": "schema", "issues": issues})
        report["parseable_assistant_json"] += 1

    report["error_samples"] = err_samples
    n = report["lines"]
    if n:
        report["parseable_rate"] = report["parseable_assistant_json"] / n
        report["schema_clean_rate"] = (report["parseable_assistant_json"] - report["schema_issues"]) / n
    else:
        report["parseable_rate"] = 0.0
        report["schema_clean_rate"] = 0.0
    return report


def evaluate_predictions_jsonl(
    predictions_path: str | Path,
    *,
    raw_field: str = "generated",
) -> dict[str, Any]:
    """Each line: JSON object with a string field (default ``generated``) to parse as structured JSON."""
    path = Path(predictions_path)
    rep: dict[str, Any] = {
        "path": str(path.resolve()),
        "lines": 0,
        "parse_ok": 0,
        "parse_fail": 0,
        "samples": [],
    }
    if not path.exists():
        rep["note"] = "file_missing"
        return rep
    samples: list[dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        rep["lines"] += 1
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            rep["parse_fail"] += 1
            continue
        raw = row.get(raw_field) if isinstance(row, dict) else None
        if not isinstance(raw, str):
            rep["parse_fail"] += 1
            continue
        obj, err = parse_assistant_json(raw)
        if err:
            rep["parse_fail"] += 1
            if len(samples) < 5:
                samples.append({"line": line_no, "error": err, "tail": raw[-120:]})
        else:
            rep["parse_ok"] += 1
    rep["samples"] = samples
    if rep["lines"]:
        rep["parse_success_rate"] = rep["parse_ok"] / rep["lines"]
    else:
        rep["parse_success_rate"] = 0.0
    return rep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate structured JSON in SFT or prediction JSONL.")
    p.add_argument("--sft-jsonl", default=None, help="Train or valid messages JSONL from build_finetune_dataset.")
    p.add_argument("--predictions-jsonl", default=None, help="Lines with {\"generated\": \"...\"} model strings.")
    p.add_argument("--output-json", default=None, help="Write combined report JSON here.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out: dict[str, Any] = {}
    if args.sft_jsonl:
        out["sft"] = evaluate_sft_jsonl(args.sft_jsonl)
    if args.predictions_jsonl:
        out["predictions"] = evaluate_predictions_jsonl(args.predictions_jsonl)
    text = json.dumps(out, indent=2, ensure_ascii=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
