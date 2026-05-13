import json
from pathlib import Path

import pytest

from build_finetune_dataset import build_chat_example, build_finetune_dataset


def test_build_chat_example_contains_chat_messages():
    record = {
        "id": "doc-1",
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

    example = build_chat_example(record, 1, max_text_chars=5000)

    assert [message["role"] for message in example["messages"]] == ["system", "user", "assistant"]
    assistant_payload = json.loads(example["messages"][2]["content"])
    assert assistant_payload["document_id"] == "doc-1"
    assert assistant_payload["course_codes"] == ["CS 101"]
    assert assistant_payload["emails"] == ["jane@example.edu"]
    assert "40%" in assistant_payload["grading_weights"]


def test_build_finetune_dataset_writes_train_and_valid_files(tmp_path: Path):
    input_path = tmp_path / "ingested.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "CS 101 syllabus\nGrading: Homework 40%\nInstructor: Professor Jane Smith"}),
                json.dumps({"id": "b", "text": "MATH 201 syllabus\nRequired Readings: Chapter 2\nWeek: Limits"}),
                json.dumps({"id": "c", "text": "BIO 110 syllabus\nAssignment 1 due 09/12/2026\nContact prof@example.edu"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    train_output = tmp_path / "finetune" / "train.jsonl"
    valid_output = tmp_path / "finetune" / "valid.jsonl"

    train, valid = build_finetune_dataset(
        input_path,
        train_output,
        valid_output,
        validation_ratio=0.34,
        seed=7,
    )

    assert train_output.exists()
    assert valid_output.exists()
    assert len(train) + len(valid) == 3
    assert len(valid) == 1
    written_train = train_output.read_text(encoding="utf-8").splitlines()
    written_valid = valid_output.read_text(encoding="utf-8").splitlines()
    assert written_train
    assert written_valid


def test_build_finetune_dataset_skips_rows_without_usable_text(tmp_path: Path):
    input_path = tmp_path / "ingested.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "empty", "text": ""}),
                json.dumps({"id": "bad", "text": "   "}),
                json.dumps(
                    {
                        "id": "good",
                        "text": "CS 101 syllabus\nGrading: Homework 40%\nInstructor: Professor Jane Smith",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_output = tmp_path / "finetune" / "train.jsonl"
    valid_output = tmp_path / "finetune" / "valid.jsonl"

    train, valid = build_finetune_dataset(input_path, train_output, valid_output, validation_ratio=0.0, seed=1)

    assert len(train) == 1
    assert len(valid) == 0
    assert json.loads(train_output.read_text(encoding="utf-8").splitlines()[0])["messages"]


def test_build_finetune_dataset_raises_when_no_usable_text(tmp_path: Path):
    input_path = tmp_path / "ingested.jsonl"
    input_path.write_text(json.dumps({"id": "x", "text": ""}) + "\n", encoding="utf-8")
    train_output = tmp_path / "finetune" / "train.jsonl"
    valid_output = tmp_path / "finetune" / "valid.jsonl"

    with pytest.raises(ValueError, match="No usable document text"):
        build_finetune_dataset(input_path, train_output, valid_output)


def test_build_finetune_dataset_allow_empty_outputs(tmp_path: Path):
    input_path = tmp_path / "ingested.jsonl"
    input_path.write_text(json.dumps({"id": "x", "text": ""}) + "\n", encoding="utf-8")
    train_output = tmp_path / "finetune" / "train.jsonl"
    valid_output = tmp_path / "finetune" / "valid.jsonl"

    train, valid = build_finetune_dataset(
        input_path,
        train_output,
        valid_output,
        allow_empty_outputs=True,
    )
    assert train == [] and valid == []
    assert train_output.read_text(encoding="utf-8") == ""
    assert valid_output.read_text(encoding="utf-8") == ""
