"""Fault-tolerant parsing and metrics for structured JSON (SFT labels or model output)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_finetune_dataset import STUDY_PLAN_LIST_FIELDS

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
        "study_plan",
        "text_field",
        "entities",
    }
)

STRING_LIST_FIELDS_FOR_METRICS = (
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
)

STUDY_PLAN_ITEM_KEYS = frozenset({"section_heading", *STUDY_PLAN_LIST_FIELDS})


def parse_assistant_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse model or dataset JSON string; return ``(obj, None)`` or ``(None, error)``."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(data, dict):
        return None, "root_not_object"
    return data, None


def _flatten_study_plan(plan: Any) -> dict[str, set[str]]:
    buckets = {field: set() for field in STUDY_PLAN_LIST_FIELDS}
    if not isinstance(plan, list):
        return buckets
    for block in plan:
        if not isinstance(block, dict):
            continue
        for field in STUDY_PLAN_LIST_FIELDS:
            buckets[field] |= _normalized_string_set(block.get(field))
    return buckets


def _normalized_string_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    out: set[str] = set()
    for item in values:
        if item is None:
            continue
        s = str(item).strip().lower()
        if s:
            out.add(s)
    return out


def f1_precision_recall(pred_set: set[str], gold_set: set[str]) -> tuple[float, float, float]:
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        # Predictions with no gold labels are false positives.
        return 0.0, 0.0, 0.0
    if not pred_set:
        return 0.0, 1.0, 0.0
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compare_assistant_payloads(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Per-field set overlap vs gold for string-list fields; ``document_id`` exact match (string)."""
    doc_pred = pred.get("document_id")
    doc_gold = gold.get("document_id")
    doc_match = doc_pred is not None and doc_gold is not None and str(doc_pred) == str(doc_gold)

    field_scores: dict[str, dict[str, float]] = {}
    macro_f1_sum = 0.0
    n_fields = 0
    for key in STRING_LIST_FIELDS_FOR_METRICS:
        f1, prec, rec = f1_precision_recall(
            _normalized_string_set(pred.get(key)),
            _normalized_string_set(gold.get(key)),
        )
        field_scores[key] = {"f1": f1, "precision": prec, "recall": rec}
        macro_f1_sum += f1
        n_fields += 1

    macro_f1 = macro_f1_sum / n_fields if n_fields else 0.0

    gold_sp = _flatten_study_plan(gold.get("study_plan"))
    pred_sp = _flatten_study_plan(pred.get("study_plan"))
    study_field_scores: dict[str, dict[str, float]] = {}
    study_sum = 0.0
    for field in STUDY_PLAN_LIST_FIELDS:
        f1, prec, rec = f1_precision_recall(pred_sp[field], gold_sp[field])
        study_field_scores[field] = {"f1": f1, "precision": prec, "recall": rec}
        study_sum += f1
    study_plan_macro_f1 = study_sum / len(STUDY_PLAN_LIST_FIELDS) if STUDY_PLAN_LIST_FIELDS else 0.0

    entities_pred = pred.get("entities")
    entities_gold = gold.get("entities")
    entities_len_match = (
        isinstance(entities_pred, list)
        and isinstance(entities_gold, list)
        and len(entities_pred) == len(entities_gold)
    )

    return {
        "document_id_match": doc_match,
        "macro_f1_string_lists": macro_f1,
        "field_f1": field_scores,
        "study_plan_macro_f1": study_plan_macro_f1,
        "study_plan_field_f1": study_field_scores,
        "entities_length_match": entities_len_match,
        "text_field_match": pred.get("text_field") == gold.get("text_field"),
    }


def aggregate_compare_stats(per_example: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_example:
        return {"examples": 0}
    n = len(per_example)
    doc_ok = sum(1 for row in per_example if row.get("document_id_match"))
    parse_ok = sum(1 for row in per_example if row.get("parsed"))
    macro_avg = sum(float(row.get("macro_f1_string_lists") or 0.0) for row in per_example) / n
    study_avg = sum(float(row.get("study_plan_macro_f1") or 0.0) for row in per_example) / n
    return {
        "examples": n,
        "json_parse_success_rate": parse_ok / n,
        "document_id_accuracy": doc_ok / n,
        "mean_macro_f1_string_lists": macro_avg,
        "mean_study_plan_macro_f1": study_avg,
    }


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
    study_plan = obj.get("study_plan")
    if study_plan is not None:
        if not isinstance(study_plan, list):
            issues.append("study_plan_not_list")
        else:
            for i, block in enumerate(study_plan):
                if not isinstance(block, dict):
                    issues.append(f"study_plan_item_{i}_not_object")
                    continue
                unknown = set(block.keys()) - STUDY_PLAN_ITEM_KEYS
                if unknown:
                    issues.append(f"study_plan_item_{i}_extra_keys:{sorted(unknown)}")
                sh = block.get("section_heading")
                if sh is not None and not isinstance(sh, str):
                    issues.append(f"study_plan_item_{i}_section_heading_not_str")
                for field in STUDY_PLAN_LIST_FIELDS:
                    vals = block.get(field)
                    if vals is not None and not isinstance(vals, list):
                        issues.append(f"study_plan_item_{i}_{field}_not_list")
                        continue
                    if isinstance(vals, list):
                        bad = False
                        for x in vals:
                            if x is not None and not isinstance(x, str):
                                bad = True
                                break
                        if bad:
                            issues.append(f"study_plan_item_{i}_{field}_non_string_entry")
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
