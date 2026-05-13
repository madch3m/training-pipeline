"""Chunked edge inference and deterministic merge logic for study-plan payloads."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from build_finetune_dataset import STUDY_PLAN_LIST_FIELDS
from process_syllabi_jsonl import SECTION_HEADERS


TOP_LEVEL_LIST_FIELDS: tuple[str, ...] = (
    "course_codes",
    "instructors",
    "emails",
    "section_names",
    "assignments",
    "readings",
    "grading_weights",
    "due_dates",
    "course_dates",
    "concepts",
)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        s = item.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def normalize_date_text(value: str) -> str:
    v = " ".join(value.replace(",", ", ").split())
    v = v.replace(" ,", ",")
    return v


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(payload)
    for field in ("due_dates", "course_dates"):
        vals = out.get(field)
        if isinstance(vals, list):
            out[field] = _dedupe_keep_order([normalize_date_text(str(v)) for v in vals if v is not None])
    for field in TOP_LEVEL_LIST_FIELDS:
        vals = out.get(field)
        if isinstance(vals, list):
            out[field] = _dedupe_keep_order([str(v) for v in vals if v is not None])

    blocks = out.get("study_plan")
    if isinstance(blocks, list):
        norm_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            nb: dict[str, Any] = {"section_heading": str(block.get("section_heading", "")).strip()}
            for field in STUDY_PLAN_LIST_FIELDS:
                vals = block.get(field)
                if isinstance(vals, list):
                    processed = [str(v) for v in vals if v is not None]
                    if field in ("due_dates", "course_dates"):
                        processed = [normalize_date_text(x) for x in processed]
                    nb[field] = _dedupe_keep_order(processed)
                else:
                    nb[field] = []
            norm_blocks.append(nb)
        out["study_plan"] = norm_blocks
    return out


def split_text_into_chunks(text: str, max_chars: int = 4000) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    lowered = text.lower()
    anchors: list[int] = []
    for header in SECTION_HEADERS:
        start = lowered.find(header)
        while start != -1:
            anchors.append(start)
            start = lowered.find(header, start + len(header))
    anchors = sorted(set(anchors))
    if not anchors:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    # Build section boundaries from anchors; then merge sections until max_chars.
    bounds = anchors + [len(text)]
    sections = [text[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    chunks: list[str] = []
    cur = ""
    for section in sections:
        if not cur:
            cur = section
            continue
        if len(cur) + len(section) <= max_chars:
            cur += section
        else:
            chunks.append(cur)
            cur = section
    if cur:
        chunks.append(cur)
    return chunks


def merge_chunk_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {
            "document_id": None,
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
            "study_plan": [],
            "text_field": "text",
            "entities": [],
        }

    normalized = [normalize_payload(p) for p in payloads]
    base = normalized[0]
    out: dict[str, Any] = {
        "document_id": base.get("document_id"),
        "source_url": base.get("source_url"),
        "text_field": base.get("text_field", "text"),
        "entities": [],
    }
    for field in TOP_LEVEL_LIST_FIELDS:
        merged: list[str] = []
        for p in normalized:
            vals = p.get(field)
            if isinstance(vals, list):
                merged.extend(str(v) for v in vals if v is not None)
        out[field] = _dedupe_keep_order(merged)

    heading_to_block: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {field: [] for field in STUDY_PLAN_LIST_FIELDS}
    )
    for p in normalized:
        blocks = p.get("study_plan")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            heading = str(block.get("section_heading", "")).strip()
            slot = heading_to_block[heading]
            for field in STUDY_PLAN_LIST_FIELDS:
                vals = block.get(field)
                if isinstance(vals, list):
                    slot[field].extend(str(v) for v in vals if v is not None)

    study_plan: list[dict[str, Any]] = []
    for heading in sorted(heading_to_block.keys(), key=lambda x: (x == "", x.lower())):
        row: dict[str, Any] = {"section_heading": heading}
        slot = heading_to_block[heading]
        for field in STUDY_PLAN_LIST_FIELDS:
            row[field] = _dedupe_keep_order(slot[field])
        if heading or any(row[field] for field in STUDY_PLAN_LIST_FIELDS):
            study_plan.append(row)
    out["study_plan"] = study_plan
    return out


@dataclass
class EdgeGenerationConfig:
    max_chars_per_chunk: int = 4000
    max_new_tokens: int = 768


def build_chunk_prompts(text: str, cfg: EdgeGenerationConfig) -> list[str]:
    prompts: list[str] = []
    for chunk in split_text_into_chunks(text, max_chars=cfg.max_chars_per_chunk):
        prompts.append(
            (
                "Extract syllabus study plan JSON using the schema keys from training "
                "(document_id, source_url, list fields, study_plan, text_field, entities). "
                "Return valid JSON only.\n\n"
                f"Document text:\n{chunk}"
            )
        )
    return prompts


def merged_payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)
