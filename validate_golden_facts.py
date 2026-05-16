"""Per-row validator for the SyllabusFacts hand-labeling file.

Loads ``data/labeled/golden_facts.jsonl``, validates each row's ``facts`` block
against the Pydantic ``SyllabusFacts`` schema, and reports:
- ``ok``       row passes Pydantic validation and has been filled in.
- ``fail``     row's facts block fails validation; prints field-level error.
- ``unfilled`` row passed validation but everything is still at template defaults
              (warning only — supports partial labeling sessions).

Also surfaces:
- coverage per important field (how many of the rows have it filled).
- grading-weight sanity (warns when sum is outside [80, 110] — wide window allows
  for "drop lowest" mechanics and partial schemes).
- due_date sanity (warns when any task's due_date year is outside [2000, 2030] —
  catches hallucinated dates that Pydantic accepts as valid years but are clearly
  invented; gpt-5 has been observed producing years like 1129).

Exit code: 0 if no ``fail`` rows; 1 otherwise. Unfilled rows do not fail the run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from syllabus_schema import MAX_DUE_YEAR, MIN_DUE_YEAR, SyllabusFacts


COVERAGE_FIELDS: tuple[str, ...] = (
    "course_code",
    "course_title",
    "instructor",
    "term",
    "class_meeting_pattern",
)


def is_unfilled(facts: dict[str, Any]) -> bool:
    """A row is unfilled if all labeler fields are still at template defaults."""
    for field in COVERAGE_FIELDS:
        if facts.get(field) is not None:
            return False
    if facts.get("grading"):
        return False
    if facts.get("all_tasks"):
        return False
    return True


def field_has_value(facts: dict[str, Any], field: str) -> bool:
    val = facts.get(field)
    if isinstance(val, list):
        return len(val) > 0
    return val is not None


def grading_weight_total(facts: dict[str, Any]) -> float:
    grading = facts.get("grading") or []
    total = 0.0
    for item in grading:
        try:
            total += float(item.get("weight_pct", 0))
        except (TypeError, ValueError):
            pass
    return total


def collect_suspicious_due_dates(
    facts: dict[str, Any],
    min_year: int = MIN_DUE_YEAR,
    max_year: int = MAX_DUE_YEAR,
) -> list[str]:
    """Return human-readable warnings for tasks whose due_date.year is out of range."""
    out: list[str] = []
    for idx, task in enumerate(facts.get("all_tasks") or []):
        if not isinstance(task, dict):
            continue
        due = task.get("due_date")
        if due is None:
            continue
        try:
            year = int(str(due)[:4])
        except (ValueError, TypeError):
            continue
        if year < min_year or year > max_year:
            title = task.get("title", "?")
            out.append(f"task[{idx}] '{title}' due_date={due} (year={year})")
    return out


def format_validation_error(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        out.append(f"{loc}: {err['msg']}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SyllabusFacts hand-labels.")
    p.add_argument("--input-jsonl", default="data/labeled/golden_facts.jsonl")
    p.add_argument("--report-json", default=None, help="Optional machine-readable report path.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-row ok lines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        raise SystemExit(f"Not found: {input_path}")

    rows: list[dict[str, Any]] = []
    parse_errors: list[tuple[int, str]] = []
    for i, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as exc:
            parse_errors.append((i, str(exc)))

    print(f"[validate] Loaded {len(rows)} rows from {input_path}")
    if parse_errors:
        for line_no, msg in parse_errors:
            print(f"[fail] line {line_no:3d}  JSON parse error: {msg}")

    n_ok = n_fail = n_unfilled = 0
    coverage = Counter()
    weight_warnings: list[tuple[int, float]] = []
    date_warnings: list[tuple[int, list[str]]] = []
    per_row_report: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        facts = row.get("facts")
        doc_id = (facts or {}).get("document_id") if isinstance(facts, dict) else None
        short_id = (doc_id[:12] + "...") if isinstance(doc_id, str) and len(doc_id) > 12 else (doc_id or "?")

        if not isinstance(facts, dict):
            n_fail += 1
            print(f"[fail] row {idx:3d}  doc={short_id}  missing or non-object 'facts' block")
            per_row_report.append({"row": idx, "status": "fail", "doc_id": doc_id, "errors": ["missing 'facts' block"]})
            continue

        try:
            SyllabusFacts.model_validate(facts)
        except ValidationError as exc:
            n_fail += 1
            errs = format_validation_error(exc)
            print(f"[fail] row {idx:3d}  doc={short_id}")
            for e in errs:
                print(f"          - {e}")
            per_row_report.append({"row": idx, "status": "fail", "doc_id": doc_id, "errors": errs})
            continue

        if is_unfilled(facts):
            n_unfilled += 1
            print(f"[skip] row {idx:3d}  doc={short_id}  unfilled (template defaults)")
            per_row_report.append({"row": idx, "status": "unfilled", "doc_id": doc_id})
            continue

        n_ok += 1
        for field in COVERAGE_FIELDS:
            if field_has_value(facts, field):
                coverage[field] += 1
        if facts.get("grading"):
            coverage["grading"] += 1
        if facts.get("all_tasks"):
            coverage["all_tasks"] += 1
        total = grading_weight_total(facts)
        if total and not (80.0 <= total <= 110.0):
            weight_warnings.append((idx, total))
        date_warns = collect_suspicious_due_dates(facts)
        if date_warns:
            date_warnings.append((idx, date_warns))

        if not args.quiet:
            label = facts.get("course_code") or facts.get("course_title") or ""
            print(f"[ok]   row {idx:3d}  doc={short_id}  {label}")
        per_row_report.append({
            "row": idx,
            "status": "ok",
            "doc_id": doc_id,
            "grading_weight_total": total,
            "date_warnings": date_warns,
        })

    total_rows = len(rows)
    print(f"\n[summary] valid={n_ok} invalid={n_fail} unfilled={n_unfilled} of {total_rows}")
    if total_rows:
        print("[coverage] " + " ".join(
            f"{field}={coverage[field]}/{total_rows}"
            for field in (*COVERAGE_FIELDS, "grading", "all_tasks")
        ))
    if weight_warnings:
        rows_str = ", ".join(f"row {r} (sum={t:.0f}%)" for r, t in weight_warnings[:10])
        print(f"[warn] grading weights outside [80, 110]: {rows_str}")
    if date_warnings:
        print(f"[warn] suspicious due_date years (outside [{MIN_DUE_YEAR}, {MAX_DUE_YEAR}]):")
        for row_idx, warns in date_warnings[:10]:
            for w in warns:
                print(f"          row {row_idx}: {w}")

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(
            json.dumps(
                {
                    "summary": {
                        "valid": n_ok,
                        "invalid": n_fail,
                        "unfilled": n_unfilled,
                        "total": total_rows,
                    },
                    "coverage": dict(coverage),
                    "weight_warnings": [{"row": r, "total": t} for r, t in weight_warnings],
                    "date_warnings": [{"row": r, "warnings": w} for r, w in date_warnings],
                    "rows": per_row_report,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    if n_fail or parse_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
