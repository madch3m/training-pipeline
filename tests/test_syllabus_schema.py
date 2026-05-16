from datetime import date

import pytest
from pydantic import ValidationError

from syllabus_schema import GradingItem, StudyTask, SyllabusFacts, StudyPlan, WeekPlan


def test_minimal_facts_has_defaults():
    f = SyllabusFacts(document_id="doc1")
    assert f.term_weeks == 15
    assert f.all_tasks == []
    assert f.grading == []
    assert f.source_url is None


def test_due_date_optional_on_task():
    t = StudyTask(title="Read Ch.1", type="reading")
    assert t.due_date is None


def test_grading_weight_bounds():
    GradingItem(category="Midterm", weight_pct=30)
    with pytest.raises(ValidationError):
        GradingItem(category="X", weight_pct=120)
    with pytest.raises(ValidationError):
        GradingItem(category="X", weight_pct=-1)


def test_extra_fields_forbidden_on_facts():
    with pytest.raises(ValidationError):
        SyllabusFacts(document_id="x", surprise="bad")


def test_extra_fields_forbidden_on_task():
    with pytest.raises(ValidationError):
        StudyTask(title="X", type="reading", priority="high")


def test_term_weeks_bounds():
    with pytest.raises(ValidationError):
        SyllabusFacts(document_id="x", term_weeks=25)
    with pytest.raises(ValidationError):
        SyllabusFacts(document_id="x", term_weeks=0)


def test_task_minutes_bounds():
    with pytest.raises(ValidationError):
        StudyTask(title="T", type="lecture", estimated_minutes=4)
    with pytest.raises(ValidationError):
        StudyTask(title="T", type="lecture", estimated_minutes=601)


def test_task_type_must_be_in_literal():
    with pytest.raises(ValidationError):
        StudyTask(title="T", type="nap")


def test_due_date_coerced_to_none_when_year_too_old():
    t = StudyTask(title="X", type="exam", due_date="1129-02-09")
    assert t.due_date is None


def test_due_date_coerced_to_none_when_year_too_new():
    t = StudyTask(title="X", type="exam", due_date="2099-01-01")
    assert t.due_date is None


def test_due_date_unchanged_when_in_range():
    t = StudyTask(title="X", type="exam", due_date="2024-09-15")
    assert t.due_date == date(2024, 9, 15)


def test_due_date_none_unchanged():
    t = StudyTask(title="X", type="reading", due_date=None)
    assert t.due_date is None


def test_due_date_coercion_inside_facts_round_trip():
    f = SyllabusFacts(
        document_id="x",
        all_tasks=[
            {"title": "Bad year", "type": "exam", "due_date": "1129-02-09"},
            {"title": "Good", "type": "exam", "due_date": "2024-09-15"},
        ],
    )
    assert f.all_tasks[0].due_date is None
    assert f.all_tasks[1].due_date == date(2024, 9, 15)


def test_studyplan_round_trips_through_json():
    f = SyllabusFacts(
        document_id="x",
        all_tasks=[StudyTask(title="HW1", type="assignment", due_date=date(2024, 1, 15))],
    )
    week = WeekPlan(
        week_number=1, start_date=date(2024, 1, 15), end_date=date(2024, 1, 21),
        tasks=f.all_tasks,
    )
    plan = StudyPlan(facts=f, inference_date=date(2024, 1, 15), weeks=[week])
    js = plan.model_dump_json()
    restored = StudyPlan.model_validate_json(js)
    assert restored == plan
