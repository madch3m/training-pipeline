import json
import subprocess
import sys
from pathlib import Path

import pytest

from validate_golden_facts import (
    collect_suspicious_due_dates,
    field_has_value,
    grading_weight_total,
    is_unfilled,
)


def _template_row():
    return {
        "facts": {
            "document_id": "x",
            "source_url": None,
            "course_code": None,
            "course_title": None,
            "instructor": None,
            "term": None,
            "term_weeks": 15,
            "class_meeting_pattern": None,
            "grading": [],
            "all_tasks": [],
        },
        "_hint": {"title": "Demo"},
    }


def _filled_row():
    row = _template_row()
    row["facts"].update(
        {
            "course_code": "CS 101",
            "course_title": "Intro to CS",
            "instructor": "Prof. X",
            "term": "Fall 2024",
            "grading": [{"category": "Final", "weight_pct": 100, "count": None}],
            "all_tasks": [{"title": "Read Ch.1", "type": "reading", "due_date": None, "estimated_minutes": None, "source_section": None}],
        }
    )
    return row


def test_is_unfilled_true_for_template():
    assert is_unfilled(_template_row()["facts"]) is True


def test_is_unfilled_false_when_any_field_filled():
    row = _template_row()
    row["facts"]["course_code"] = "X"
    assert is_unfilled(row["facts"]) is False


def test_is_unfilled_false_when_grading_present():
    row = _template_row()
    row["facts"]["grading"] = [{"category": "X", "weight_pct": 50}]
    assert is_unfilled(row["facts"]) is False


def test_field_has_value_handles_lists():
    facts = _template_row()["facts"]
    assert field_has_value(facts, "course_code") is False
    assert field_has_value(facts, "all_tasks") is False
    facts["all_tasks"] = [{"title": "T", "type": "reading"}]
    assert field_has_value(facts, "all_tasks") is True


def test_grading_weight_total_sums_correctly():
    facts = {
        "grading": [
            {"category": "A", "weight_pct": 30},
            {"category": "B", "weight_pct": 70},
        ]
    }
    assert grading_weight_total(facts) == 100.0


def test_grading_weight_total_skips_garbage():
    facts = {"grading": [{"category": "A", "weight_pct": "garbage"}, {"category": "B", "weight_pct": 25}]}
    assert grading_weight_total(facts) == 25.0


def test_collect_suspicious_due_dates_no_tasks():
    assert collect_suspicious_due_dates({}) == []


def test_collect_suspicious_due_dates_in_range():
    facts = {"all_tasks": [{"title": "X", "type": "exam", "due_date": "2024-09-01"}]}
    assert collect_suspicious_due_dates(facts) == []


def test_collect_suspicious_due_dates_ignores_null_dates():
    facts = {"all_tasks": [{"title": "X", "type": "reading", "due_date": None}]}
    assert collect_suspicious_due_dates(facts) == []


def test_collect_suspicious_due_dates_flags_year_too_old():
    facts = {"all_tasks": [{"title": "Old Exam", "type": "exam", "due_date": "1129-02-09"}]}
    warns = collect_suspicious_due_dates(facts)
    assert len(warns) == 1
    assert "1129" in warns[0] and "Old Exam" in warns[0]


def test_collect_suspicious_due_dates_flags_year_too_new():
    facts = {"all_tasks": [
        {"title": "Future", "type": "exam", "due_date": "2099-01-01"},
        {"title": "OK", "type": "reading", "due_date": "2025-09-01"},
    ]}
    warns = collect_suspicious_due_dates(facts)
    assert len(warns) == 1 and "Future" in warns[0]


def test_collect_suspicious_due_dates_custom_window():
    facts = {"all_tasks": [{"title": "X", "type": "exam", "due_date": "2025-01-01"}]}
    assert collect_suspicious_due_dates(facts, min_year=2026, max_year=2030) == [
        "task[0] 'X' due_date=2025-01-01 (year=2025)"
    ]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows), encoding="utf-8")


def test_cli_exits_zero_on_all_filled(tmp_path):
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, [_filled_row(), _filled_row()])
    result = subprocess.run(
        [sys.executable, "validate_golden_facts.py", "--input-jsonl", str(p), "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid=2" in result.stdout
    assert "invalid=0" in result.stdout


def test_cli_exits_nonzero_on_invalid(tmp_path):
    bad = _filled_row()
    bad["facts"]["all_tasks"][0]["type"] = "homework"  # not in Literal
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, [bad])
    result = subprocess.run(
        [sys.executable, "validate_golden_facts.py", "--input-jsonl", str(p), "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "invalid=1" in result.stdout
    assert "all_tasks.0.type" in result.stdout


def test_cli_treats_unfilled_as_skip_not_fail(tmp_path):
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, [_template_row(), _filled_row()])
    result = subprocess.run(
        [sys.executable, "validate_golden_facts.py", "--input-jsonl", str(p), "--quiet"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "unfilled=1" in result.stdout
    assert "valid=1" in result.stdout


def test_cli_warns_on_suspicious_year_but_does_not_fail(tmp_path):
    bad = _filled_row()
    bad["facts"]["all_tasks"][0]["type"] = "exam"
    bad["facts"]["all_tasks"][0]["due_date"] = "1129-02-09"
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, [bad])
    result = subprocess.run(
        [sys.executable, "validate_golden_facts.py", "--input-jsonl", str(p), "--quiet"],
        capture_output=True, text=True,
    )
    # Pydantic accepts year 1129 as a valid date — heuristic warning, not failure.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[warn] suspicious due_date" in result.stdout
    assert "1129" in result.stdout
    assert "valid=1" in result.stdout
