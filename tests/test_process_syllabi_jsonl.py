import json
from pathlib import Path

from process_syllabi_jsonl import (
    build_output_record,
    detect_text_field,
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
