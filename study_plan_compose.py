"""Deterministic StudyPlan composition from SyllabusFacts + inference date.

Rules:
- Tasks with ``due_date`` go in the week containing that date (clamped to term).
- Tasks without ``due_date`` are distributed uniformly across all weeks (round-robin
  in their original order).
- Week 1 starts on the Monday of the week containing ``inference_date``; each
  subsequent week starts 7 days later.
"""

from __future__ import annotations

from datetime import date, timedelta

from syllabus_schema import StudyPlan, StudyTask, SyllabusFacts, WeekPlan


def _week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _sum_minutes(tasks: list[StudyTask]) -> int | None:
    minutes = [t.estimated_minutes for t in tasks if t.estimated_minutes is not None]
    return sum(minutes) if minutes else None


def compose_study_plan(facts: SyllabusFacts, inference_date: date) -> StudyPlan:
    term_weeks = facts.term_weeks
    week1_start = _week_start_monday(inference_date)
    week_starts = [week1_start + timedelta(weeks=i) for i in range(term_weeks)]

    by_week: dict[int, list[StudyTask]] = {i: [] for i in range(term_weeks)}
    unanchored: list[StudyTask] = []
    for task in facts.all_tasks:
        if task.due_date is None:
            unanchored.append(task)
            continue
        delta_days = (task.due_date - week1_start).days
        idx = max(0, min(term_weeks - 1, delta_days // 7))
        by_week[idx].append(task)

    for i, task in enumerate(unanchored):
        by_week[i % term_weeks].append(task)

    weeks = [
        WeekPlan(
            week_number=i + 1,
            start_date=week_starts[i],
            end_date=week_starts[i] + timedelta(days=6),
            tasks=by_week[i],
            estimated_total_minutes=_sum_minutes(by_week[i]),
        )
        for i in range(term_weeks)
    ]
    return StudyPlan(facts=facts, inference_date=inference_date, weeks=weeks)
