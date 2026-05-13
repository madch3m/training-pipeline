import json
from pathlib import Path

import pytest

from process_syllabi_jsonl import (
    build_output_record,
    detect_text_field,
    has_usable_document_text,
    load_jsonl,
    process_jsonl,
)


def test_load_jsonl_reads_multiple_rows(tmp_path: Path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"id":"a","text":"one"}\n{"id":"b","text":"two"}\n', encoding="utf-8")
    rows = load_jsonl(path)
    assert [row["id"] for row in rows] == ["a", "b"]


def test_detect_text_field_prefers_known_text_keys():
    field, value = detect_text_field({"content": "short", "text": "syllabus body"})
    assert field == "text"
    assert value == "syllabus body"


def test_build_output_record_runs_regex_and_ner():
    record = {
        "id": "syllabus_1",
        "source_url": "https://example.edu/syllabus",
        "text": (
            "CS 101 Syllabus\n"
            "Instructor: Professor Jane Smith\n"
            "Office Hours Tuesday 1-3pm\n"
            "Grading: Homework 40%\n"
            "Assignment 1 due Sep 15, 2026\n"
            "Required Readings: Chapter 1\n"
            "Week: Regular Expressions\n"
            "Contact jane@example.edu\n"
        ),
    }

    output = build_output_record(record, 1)
    regex_labels = {item["label"] for item in output["regex_entities"]}
    ner_labels = {item["label"] for item in output["ner_entities"]}

    assert "COURSE" in regex_labels
    assert "GRADING_WEIGHT" in regex_labels
    assert "DUE_DATE" in regex_labels
    assert "READING" in regex_labels
    assert "EMAIL" in regex_labels
    assert "INSTRUCTOR" in ner_labels
    assert "SECTION" in ner_labels
    assert "CONCEPT" in ner_labels


def test_has_usable_document_text_false_for_empty_text():
    assert not has_usable_document_text({"id": "1", "text": ""})
    assert not has_usable_document_text({"id": "1", "text": "   "})
    assert has_usable_document_text({"id": "1", "text": "hello world syllabus body " * 5})


def test_process_jsonl_skips_rows_without_usable_text(tmp_path: Path):
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "labeled" / "entities.jsonl"
    input_path.write_text(
        json.dumps({"doc_id": "empty", "text": ""}) + "\n"
        + json.dumps(
            {"doc_id": "ok", "body": "EECS 201 syllabus\nGrading: Project 25%\nRequired Readings: paper one"},
        )
        + "\n",
        encoding="utf-8",
    )

    processed = process_jsonl(input_path, output_path)

    assert len(processed) == 1
    assert processed[0]["id"] == "ok"


def test_load_jsonl_strict_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(path, strict=True)


def test_load_jsonl_tolerant_skips_bad_lines(tmp_path: Path):
    path = tmp_path / "mix.jsonl"
    err = tmp_path / "err.jsonl"
    path.write_text('{"id":"a","text":"x"}\nnot json\n{"id":"b","text":"y"}\n', encoding="utf-8")
    rows = load_jsonl(path, strict=False, error_log_path=err)
    assert len(rows) == 2
    assert err.read_text(encoding="utf-8").strip()


def test_process_jsonl_tolerant_returns_empty_when_no_usable_rows(tmp_path: Path):
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "labeled" / "entities.jsonl"
    input_path.write_text(json.dumps({"id": "only-empty", "text": ""}) + "\n", encoding="utf-8")

    processed = process_jsonl(input_path, output_path, tolerant=True)
    assert processed == []
    assert output_path.read_text(encoding="utf-8") == ""


def test_process_jsonl_raises_when_no_usable_rows_strict_mode(tmp_path: Path):
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "labeled" / "entities.jsonl"
    input_path.write_text(json.dumps({"id": "only-empty", "text": ""}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No usable document text"):
        process_jsonl(input_path, output_path, tolerant=False)


def test_process_jsonl_tolerant_skips_invalid_json_line(tmp_path: Path):
    good = {"id": "g", "text": "EECS 201 syllabus\nGrading: Project 25%\nRequired Readings: paper one"}
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps(good) + "\nNOT JSON\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    processed = process_jsonl(input_path, output_path, tolerant=True)
    assert len(processed) == 1
    assert processed[0]["id"] == "g"


def test_process_jsonl_writes_enriched_output(tmp_path: Path):
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "labeled" / "entities.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "body": "EECS 201 syllabus\nGrading: Project 25%\nRequired Readings: paper one",
            }
        ) + "\n",
        encoding="utf-8",
    )

    processed = process_jsonl(input_path, output_path)

    assert len(processed) == 1
    assert output_path.exists()
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["id"] == "doc-1"
    assert written[0]["text_field"] == "body"
    assert written[0]["regex_entities"]
