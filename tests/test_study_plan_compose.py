from datetime import date

from study_plan_compose import compose_study_plan
from syllabus_schema import StudyTask, SyllabusFacts


def test_default_term_15_weeks():
    facts = SyllabusFacts(document_id="x")
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    assert len(plan.weeks) == 15
    assert plan.weeks[0].week_number == 1
    assert plan.weeks[-1].week_number == 15


def test_inference_date_rounds_to_monday():
    # Wed 2024-01-03 → week 1 starts Mon 2024-01-01
    facts = SyllabusFacts(document_id="x", term_weeks=2)
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 3))
    assert plan.weeks[0].start_date == date(2024, 1, 1)
    assert plan.weeks[0].end_date == date(2024, 1, 7)
    assert plan.weeks[1].start_date == date(2024, 1, 8)
    assert plan.weeks[1].end_date == date(2024, 1, 14)


def test_anchored_task_lands_in_due_date_week():
    # Inference Mon 2024-01-01; due Tue 2024-01-09 → week 2 (Jan 8–14)
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=3,
        all_tasks=[StudyTask(title="HW1", type="assignment", due_date=date(2024, 1, 9))],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    assert [len(w.tasks) for w in plan.weeks] == [0, 1, 0]
    assert plan.weeks[1].tasks[0].title == "HW1"


def test_unanchored_tasks_distributed_uniformly():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=3,
        all_tasks=[StudyTask(title=f"R{i}", type="reading") for i in range(6)],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    assert [len(w.tasks) for w in plan.weeks] == [2, 2, 2]


def test_anchored_outside_term_clamps_to_last_week():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=4,
        all_tasks=[StudyTask(title="Final", type="exam", due_date=date(2026, 1, 1))],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    assert [len(w.tasks) for w in plan.weeks] == [0, 0, 0, 1]


def test_anchored_before_term_clamps_to_first_week():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=4,
        all_tasks=[StudyTask(title="Stale", type="reading", due_date=date(2020, 1, 1))],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    assert [len(w.tasks) for w in plan.weeks] == [1, 0, 0, 0]


def test_estimated_minutes_summed_per_week():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=2,
        all_tasks=[
            StudyTask(title="A", type="reading", estimated_minutes=30),
            StudyTask(title="B", type="reading", estimated_minutes=45),
        ],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    # Round-robin → A in week 1, B in week 2
    assert plan.weeks[0].estimated_total_minutes == 30
    assert plan.weeks[1].estimated_total_minutes == 45


def test_week_with_no_minutes_is_none_not_zero():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=2,
        all_tasks=[StudyTask(title="A", type="reading")],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    # A is unanchored → goes to week 1; week 2 stays empty
    assert plan.weeks[0].estimated_total_minutes is None  # task has no estimate
    assert plan.weeks[1].estimated_total_minutes is None


def test_mixed_anchored_and_unanchored():
    facts = SyllabusFacts(
        document_id="x",
        term_weeks=3,
        all_tasks=[
            StudyTask(title="HW1", type="assignment", due_date=date(2024, 1, 16)),  # week 3 (Jan 15–21)
            StudyTask(title="R1", type="reading"),
            StudyTask(title="R2", type="reading"),
            StudyTask(title="R3", type="reading"),
        ],
    )
    plan = compose_study_plan(facts, inference_date=date(2024, 1, 1))
    # HW1 in week 3, then 3 unanchored round-robin into weeks 1/2/3
    assert [len(w.tasks) for w in plan.weeks] == [1, 1, 2]
