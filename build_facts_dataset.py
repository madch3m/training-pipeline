"""Build SFT chat examples for the SyllabusFacts extraction LoRA.

Each row is a `messages` list with three turns:
  system    : the extraction instruction (matches the labeler prompt).
  user      : raw text + regex/heuristic pre-pass features as in-context hints.
  assistant : SyllabusFacts JSON (validated against the Pydantic schema).

Inputs:
  --text-jsonl    : ingested text JSONL (raw documents).
  --labels-jsonl  : output of merge_teacher_labels.py (one row per document).

Output: train + valid JSONL pair under data/finetune_facts/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from process_syllabi_jsonl import (
    detect_text_field,
    extract_heuristic_ner,
    extract_regex_entities,
    has_usable_document_text,
    normalize_record_id,
)
from syllabus_schema import SyllabusFacts


SYSTEM_PROMPT = (
    "You extract structured facts from university syllabus documents. "
    "Return JSON conforming exactly to the SyllabusFacts schema. "
    "Set due_date ONLY when the syllabus explicitly states a date — never infer or guess. "
    "Set fields to null when the document does not state them."
)


def build_user_prompt(
    text: str,
    pre_pass_entities: list[dict[str, Any]],
    sections: list[str],
    doc_id: str,
    source_url: str | None,
) -> str:
    pre_pass = {
        "regex_entities": pre_pass_entities[:80],
        "section_headers_detected": sections,
        "document_id": doc_id,
        "source_url": source_url,
    }
    return (
        "Extract SyllabusFacts from this syllabus.\n\n"
        f"Pre-pass features (regex/heuristic, may be noisy):\n"
        f"{json.dumps(pre_pass, ensure_ascii=True)}\n\n"
        f"Document text:\n{text}\n\n"
        "Return JSON only."
    )


def index_text_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Map document_id → raw-text record (filtering rows without usable text)."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not has_usable_document_text(row):
            continue
        out[normalize_record_id(row, i)] = row
    return out


def build_chat_example(
    record: dict[str, Any],
    facts: dict[str, Any],
    max_text_chars: int,
    doc_id: str,
) -> dict[str, Any] | None:
    _, text = detect_text_field(record)
    text = text[:max_text_chars]
    regex_entities = [e.__dict__ for e in extract_regex_entities(text)]
    ner_entities = [e.__dict__ for e in extract_heuristic_ner(text)]
    sections = sorted({e["text"] for e in ner_entities if e["label"] == "SECTION"})
    user_prompt = build_user_prompt(text, regex_entities + ner_entities, sections, doc_id, record.get("source_url"))

    try:
        validated = SyllabusFacts.model_validate(facts)
    except Exception:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(validated.model_dump(mode="json"), ensure_ascii=True)},
        ]
    }


def split_examples(examples: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = int(len(shuffled) * validation_ratio)
    if validation_ratio > 0 and n_valid == 0 and len(shuffled) > 1:
        n_valid = 1
    return shuffled[n_valid:], shuffled[:n_valid]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT chat examples for SyllabusFacts extraction.")
    p.add_argument("--text-jsonl", default="data/ingested/us_freshman_core_syllabi_with_text.jsonl")
    p.add_argument("--labels-jsonl", default="data/labeled/teacher_labels.jsonl")
    p.add_argument("--train-output", default="data/finetune_facts/train.jsonl")
    p.add_argument("--valid-output", default="data/finetune_facts/valid.jsonl")
    p.add_argument("--max-text-chars", type=int, default=12_000)
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text_index = index_text_jsonl(Path(args.text_jsonl))

    examples: list[dict[str, Any]] = []
    n_missing_text = n_invalid = 0
    for line in Path(args.labels_jsonl).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        facts = row.get("facts")
        if not isinstance(facts, dict):
            continue
        doc_id = facts.get("document_id")
        if not doc_id:
            continue
        record = text_index.get(str(doc_id))
        if record is None:
            n_missing_text += 1
            continue
        ex = build_chat_example(record, facts, args.max_text_chars, str(doc_id))
        if ex is None:
            n_invalid += 1
            continue
        examples.append(ex)

    train, valid = split_examples(examples, args.validation_ratio, args.seed)
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.valid_output), valid)
    print(
        f"[build_facts] examples={len(examples)} train={len(train)} valid={len(valid)} | "
        f"skipped: missing_text={n_missing_text} invalid_facts={n_invalid}"
    )


if __name__ == "__main__":
    main()
