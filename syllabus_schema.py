"""Pydantic v2 schemas for the syllabus → study-plan pipeline.

Two qualitatively different stages:
- ``SyllabusFacts`` is the LoRA model's output: anchored facts read from a syllabus.
  Set ``due_date`` only when explicit; never infer.
- ``StudyPlan`` is composed deterministically from ``(SyllabusFacts, inference_date)``
  by ``study_plan_compose.compose_study_plan``. The model never produces ``WeekPlan``
  directly — pacing is a fixed Python function so training data stays consistent.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal[
    "reading",
    "assignment",
    "exam",
    "quiz",
    "project",
    "lecture",
    "lab",
    "discussion",
]


class StudyTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    type: TaskType
    due_date: Optional[date] = None
    estimated_minutes: Optional[int] = Field(default=None, ge=5, le=600)
    source_section: Optional[str] = None


class GradingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    weight_pct: float = Field(ge=0, le=100)
    count: Optional[int] = Field(default=None, ge=1)


class SyllabusFacts(BaseModel):
    """Anchored facts the LoRA extracts from a syllabus document.

    Default ``term_weeks=15`` applies when the syllabus does not state term length;
    see study_plan_compose for the pacing rule that uses it.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_url: Optional[str] = None
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    instructor: Optional[str] = None
    term: Optional[str] = None
    term_weeks: int = Field(default=15, ge=1, le=20)
    class_meeting_pattern: Optional[str] = None
    grading: list[GradingItem] = Field(default_factory=list)
    all_tasks: list[StudyTask] = Field(default_factory=list)


class WeekPlan(BaseModel):
    """One inferred week of study. Composed deterministically; never produced by the LoRA."""

    model_config = ConfigDict(extra="forbid")

    week_number: int = Field(ge=1, le=20)
    start_date: date
    end_date: date
    learning_objectives: list[str] = Field(default_factory=list)
    tasks: list[StudyTask] = Field(default_factory=list)
    estimated_total_minutes: Optional[int] = None


class StudyPlan(BaseModel):
    """Final composed output: facts + inferred schedule keyed to inference_date."""

    model_config = ConfigDict(extra="forbid")

    facts: SyllabusFacts
    inference_date: date
    weeks: list[WeekPlan]
    advice: Optional[str] = None
