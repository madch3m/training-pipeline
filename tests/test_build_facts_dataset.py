import json
from pathlib import Path

from build_facts_dataset import (
    SYSTEM_PROMPT,
    build_chat_example,
    index_text_jsonl,
    split_examples,
)


def test_build_chat_example_validates_facts(tmp_path):
    record = {
        "id": "doc-A",
        "source_url": "https://example.edu/syllabus",
        "text": "Course CS101 - Intro\nProfessor Smith\nMidterm: Oct 15.\n" * 10,
    }
    facts = {
        "document_id": "doc-A",
        "course_code": "CS101",
        "instructor": "Prof. Smith",
        "term_weeks": 15,
        "all_tasks": [{"title": "Midterm", "type": "exam", "due_date": "2024-10-15"}],
    }
    example = build_chat_example(record, facts, max_text_chars=2000, doc_id="doc-A")
    assert example is not None
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][0]["content"] == SYSTEM_PROMPT
    assert example["messages"][1]["role"] == "user"
    assert "Pre-pass features" in example["messages"][1]["content"]
    assert example["messages"][2]["role"] == "assistant"
    parsed = json.loads(example["messages"][2]["content"])
    assert parsed["course_code"] == "CS101"
    assert parsed["all_tasks"][0]["title"] == "Midterm"


def test_build_chat_example_returns_none_for_invalid_facts():
    record = {"id": "x", "text": "lots of words " * 50}
    bad_facts = {"document_id": "x", "term_weeks": 99}  # exceeds le=20
    assert build_chat_example(record, bad_facts, max_text_chars=500, doc_id="x") is None


def test_index_text_jsonl_skips_unparseable_and_empty(tmp_path):
    p = tmp_path / "text.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"id": "A", "text": "real " * 50}),
            "{not json",
            json.dumps({"id": "B"}),  # no usable text
            json.dumps({"id": "C", "text": "more " * 50}),
        ]),
        encoding="utf-8",
    )
    out = index_text_jsonl(p)
    assert set(out) == {"A", "C"}


def test_split_examples_respects_ratio_and_minimum(tmp_path):
    rows = [{"messages": [{"role": "system", "content": "x"}]} for _ in range(10)]
    train, valid = split_examples(rows, validation_ratio=0.2, seed=7)
    assert len(train) == 8 and len(valid) == 2

    # tiny set with non-zero ratio: at least one valid example if possible
    train, valid = split_examples(rows[:3], validation_ratio=0.05, seed=7)
    assert len(valid) == 1 and len(train) == 2
