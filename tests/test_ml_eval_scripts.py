import json
from pathlib import Path

from build_finetune_dataset import STUDY_PLAN_LIST_FIELDS
from evaluate_structured_json import (
    compare_assistant_payloads,
    evaluate_sft_jsonl,
    f1_precision_recall,
    parse_assistant_json,
)
from generate_pipeline_data_report import build_report, count_nonempty_lines, document_ids_from_sft
from pipeline_runner import SyllabusPipelineConfig


def _minimal_study_plan() -> list[dict]:
    return [{"section_heading": "", **{f: [] for f in STUDY_PLAN_LIST_FIELDS}}]
def test_parse_assistant_json_valid():
    obj, err = parse_assistant_json('{"document_id":"1","course_codes":[]}')
    assert err is None and obj is not None
    assert obj["document_id"] == "1"


def test_parse_assistant_json_invalid():
    obj, err = parse_assistant_json("not json")
    assert obj is None and err is not None


def test_f1_precision_recall_empty_sets():
    f1, p, r = f1_precision_recall(set(), set())
    assert (f1, p, r) == (1.0, 1.0, 1.0)


def test_f1_precision_recall_hallucinated_when_gold_empty():
    f1, p, r = f1_precision_recall({"a"}, set())
    assert f1 == 0.0 and p == 0.0 and r == 0.0


def test_compare_assistant_payloads_perfect():
    payload = {
        "document_id": "d1",
        "source_url": None,
        "course_codes": ["CS101"],
        "instructors": [],
        "emails": [],
        "section_names": [],
        "assignments": [],
        "readings": [],
        "grading_weights": [],
        "due_dates": [],
        "course_dates": [],
        "concepts": [],
        "study_plan": _minimal_study_plan(),
        "text_field": "text",
        "entities": [],
    }
    m = compare_assistant_payloads(payload, payload)
    assert m["document_id_match"] is True
    assert m["macro_f1_string_lists"] == 1.0
    assert m["field_f1"]["course_codes"]["f1"] == 1.0
    assert m["study_plan_macro_f1"] == 1.0


def test_evaluate_sft_jsonl_on_sample(tmp_path: Path):
    rec = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "document_id": "d1",
                        "source_url": None,
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
                        "study_plan": _minimal_study_plan(),
                        "text_field": "text",
                        "entities": [],
                    },
                    ensure_ascii=True,
                ),
            },
        ]
    }
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    rep = evaluate_sft_jsonl(path)
    assert rep["lines"] == 1
    assert rep["parseable_assistant_json"] == 1
    assert rep["assistant_json_errors"] == 0


def test_train_valid_overlap_detects_duplicate_doc_id(tmp_path: Path):
    def line(doc_id: str) -> str:
        payload = {
            "document_id": doc_id,
            "source_url": None,
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
            "study_plan": _minimal_study_plan(),
            "text_field": "text",
            "entities": [],
        }
        return json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": json.dumps(payload, ensure_ascii=True)},
                ]
            },
            ensure_ascii=True,
        )

    train_p = tmp_path / "train.jsonl"
    valid_p = tmp_path / "valid.jsonl"
    train_p.write_text(line("same") + "\n", encoding="utf-8")
    valid_p.write_text(line("same") + "\n", encoding="utf-8")
    train_ids = document_ids_from_sft(train_p)
    assert train_ids == ["same"]
    from generate_pipeline_data_report import train_valid_overlap

    o = train_valid_overlap(train_p, valid_p)
    assert o["overlap_count"] == 1


def test_build_report_smoke(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "us_freshman_core_syllabi_index.csv").write_text(
        "source_url,domain\nhttps://x.edu/a,x.edu\n",
        encoding="utf-8",
    )
    (repo / "build_finetune_dataset.py").write_text("# stub\n", encoding="utf-8")
    cfg = SyllabusPipelineConfig.for_repo(repo)
    rep = build_report(cfg)
    assert "line_counts" in rep
    assert rep["line_counts"]["url_jsonl"] == 0
    assert count_nonempty_lines(cfg.index_csv) >= 1
