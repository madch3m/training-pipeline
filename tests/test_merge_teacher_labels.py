import json
from pathlib import Path

from merge_teacher_labels import labels_agree, load_labels, merge


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows), encoding="utf-8")


def test_load_labels_returns_latest_per_doc(tmp_path):
    p = tmp_path / "labels.jsonl"
    _write(p, [
        {"facts": {"document_id": "A", "course_code": "CS101"}},
        {"facts": {"document_id": "B", "course_code": "MATH101"}},
        {"facts": {"document_id": "A", "course_code": "CS101-V2"}},  # later A wins
        {"error": "junk"},  # non-facts row ignored
    ])
    out = load_labels(p)
    assert out == {"A": {"document_id": "A", "course_code": "CS101-V2"}, "B": {"document_id": "B", "course_code": "MATH101"}}


def test_labels_agree_ignores_list_order():
    a = {"all_tasks": [{"title": "B"}, {"title": "A"}]}
    b = {"all_tasks": [{"title": "A"}, {"title": "B"}]}
    assert labels_agree(a, b)


def test_labels_disagree_on_value_change():
    a = {"course_code": "CS101"}
    b = {"course_code": "CS102"}
    assert not labels_agree(a, b)


def test_merge_priority_golden_first():
    ds = {"X": {"document_id": "X", "course_code": "DS"}}
    gt = {"X": {"document_id": "X", "course_code": "OAI"}}
    gd = {"X": {"document_id": "X", "course_code": "HUMAN"}}
    rows, stats = merge(ds, gt, gd)
    assert len(rows) == 1
    assert rows[0]["source"] == "golden"
    assert rows[0]["facts"]["course_code"] == "HUMAN"
    assert stats["golden"] == 1


def test_merge_agreement_when_deepseek_and_gpt4o_match():
    ds = {"X": {"document_id": "X", "course_code": "CS101"}}
    gt = {"X": {"document_id": "X", "course_code": "CS101"}}
    rows, stats = merge(ds, gt, {})
    assert rows[0]["source"] == "agreement"
    assert stats["agreement"] == 1


def test_merge_arbitration_keeps_gpt4o_when_disagree():
    ds = {"X": {"document_id": "X", "course_code": "CS101"}}
    gt = {"X": {"document_id": "X", "course_code": "CS101A"}}
    rows, stats = merge(ds, gt, {})
    assert rows[0]["source"] == "gpt4o_arbitrated"
    assert rows[0]["facts"]["course_code"] == "CS101A"
    assert stats["gpt4o_arbitrated"] == 1


def test_merge_falls_back_to_each_source_alone():
    ds = {"A": {"document_id": "A"}}
    gt = {"B": {"document_id": "B"}}
    rows, stats = merge(ds, gt, {})
    sources = sorted(r["source"] for r in rows)
    assert sources == ["deepseek_only", "gpt4o_only"]
    assert stats["deepseek_only"] == 1 and stats["gpt4o_only"] == 1
