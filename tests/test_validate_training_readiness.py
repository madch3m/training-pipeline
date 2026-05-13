import json
from pathlib import Path

from validate_training_readiness import ReadinessThresholds, evaluate_readiness


def _line(doc_id: str, source_url: str) -> str:
    payload = {
        "document_id": doc_id,
        "source_url": source_url,
        "course_codes": [],
        "instructors": [],
        "emails": [],
        "section_names": [],
        "assignments": [],
        "readings": [],
        "grading_weights": [],
        "due_dates": [],
        "course_dates": [],
        "concepts": [],
        "study_plan": [
            {
                "section_heading": "",
                "course_codes": [],
                "readings": [],
                "assignments": [],
                "due_dates": [],
                "course_dates": [],
                "concepts": [],
                "grading_weights": [],
                "instructors": [],
                "emails": [],
            }
        ],
        "text_field": "text",
        "entities": [],
    }
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)},
            ]
        },
        ensure_ascii=True,
    )


def test_readiness_passes_on_clean_split(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(_line("a", "https://alpha.edu/syllabus") + "\n", encoding="utf-8")
    valid.write_text(_line("b", "https://beta.edu/syllabus") + "\n", encoding="utf-8")
    rep = evaluate_readiness(train, valid, ReadinessThresholds(max_domain_overlap_ratio=0.0))
    assert rep["ok"] is True
    assert rep["violations"] == []


def test_readiness_fails_on_docid_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(_line("same", "https://alpha.edu/syllabus") + "\n", encoding="utf-8")
    valid.write_text(_line("same", "https://beta.edu/syllabus") + "\n", encoding="utf-8")
    rep = evaluate_readiness(train, valid, ReadinessThresholds(max_document_id_overlap=0))
    assert rep["ok"] is False
    assert "doc_id_overlap_ok" in rep["violations"]
