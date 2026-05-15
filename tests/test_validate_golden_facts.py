import json
import subprocess
import sys
from pathlib import Path

import pytest

from validate_golden_facts import (
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
