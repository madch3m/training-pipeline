"""Generate a hand-labeling template for SyllabusFacts (the golden set).

Picks N documents from the ingested text JSONL, stratified round-robin by school
so the labeler sees institutional variety. Each output row prefills
``document_id`` and ``source_url`` and leaves the rest of ``SyllabusFacts`` as
nulls / empty lists / the term_weeks=15 default. A sibling ``_hint`` field
carries title/school/department/subject_area for orientation; ``_hint`` lives
outside ``facts`` so it doesn't trip Pydantic's ``extra='forbid'`` and is
ignored by ``merge_teacher_labels.load_labels``.

Hand-label by editing each row in place, then save as
``data/labeled/golden_facts.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from process_syllabi_jsonl import has_usable_document_text, load_jsonl, normalize_record_id


def stratified_pick(records: list[tuple[str, dict]], n: int) -> list[tuple[str, dict]]:
    """Round-robin from `(doc_id, record)` pairs grouped by school. Stable order within group."""
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for pair in records:
        school = str(pair[1].get("school") or pair[1].get("domain") or "_unknown")
        groups[school].append(pair)

    picked: list[tuple[str, dict]] = []
    keys = list(groups.keys())
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if not groups[k]:
                continue
            if len(picked) >= n:
                break
            picked.append(groups[k].pop(0))
    return picked


def skeleton_row(doc_id: str, record: dict) -> dict[str, Any]:
    return {
        "facts": {
            "document_id": doc_id,
            "source_url": record.get("source_url"),
            "course_code": None,
            "course_title": None,
            "instructor": None,
            "term": None,
            "term_weeks": 15,
            "class_meeting_pattern": None,
            "grading": [],
            "all_tasks": [],
        },
        "_hint": {
            "title": record.get("title"),
            "school": record.get("school"),
            "department": record.get("department"),
            "subject_area": record.get("subject_area"),
            "file_type": record.get("file_type"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SyllabusFacts hand-labeling template.")
    p.add_argument("--input-jsonl", default="data/ingested/us_freshman_core_syllabi_with_text.jsonl")
    p.add_argument("--output-jsonl", default="data/labeled/golden_facts_template.jsonl")
    p.add_argument("--n", type=int, default=50, help="Number of documents to include.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl, strict=False)
    candidates: list[tuple[str, dict]] = []
    for i, record in enumerate(rows, 1):
        if not isinstance(record, dict) or not has_usable_document_text(record):
            continue
        candidates.append((normalize_record_id(record, i), record))

    if not candidates:
        raise SystemExit(f"No usable documents in {args.input_jsonl}")

    picked = stratified_pick(candidates, args.n)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for doc_id, record in picked:
            fh.write(json.dumps(skeleton_row(doc_id, record), ensure_ascii=True) + "\n")

    schools = defaultdict(int)
    for _, record in picked:
        schools[str(record.get("school") or record.get("domain") or "_unknown")] += 1
    print(f"[template] Wrote {len(picked)} skeleton rows → {out}")
    print(f"[template] School distribution: {dict(sorted(schools.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
