from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from pipeline_errors import log_pipeline_error
from process_syllabi_jsonl import (
    SECTION_HEADERS,
    detect_text_field,
    extract_heuristic_ner,
    extract_regex_entities,
    has_usable_document_text,
    load_jsonl,
    normalize_record_id,
)

SYSTEM_PROMPT = (
    "You extract structured syllabus data from university course documents. "
    "Return valid JSON only, matching the requested schema exactly."
)


def unique_texts(items: Iterable[dict], label: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        if item.get("label") != label:
            continue
        text = str(item.get("text", "")).strip()
        normalized = text.lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        values.append(text)
    return values


def section_names_from_entities(entities: Iterable[dict]) -> list[str]:
    values = unique_texts(entities, "SECTION")
    ordered: list[str] = []
    seen: set[str] = set()
    for canonical in SECTION_HEADERS:
        for value in values:
            if value.lower() == canonical and value.lower() not in seen:
                ordered.append(value)
                seen.add(value.lower())
    for value in values:
        lowered = value.lower()
        if lowered not in seen:
            ordered.append(value)
            seen.add(lowered)
    return ordered


def build_structured_target(record: dict, index: int) -> dict:
    text_field, text = detect_text_field(record)
    regex_entities = [entity.__dict__ for entity in extract_regex_entities(text)]
    ner_entities = [entity.__dict__ for entity in extract_heuristic_ner(text)]
    all_entities = regex_entities + ner_entities

    return {
        "document_id": normalize_record_id(record, index),
        "source_url": record.get("source_url"),
        "course_codes": unique_texts(regex_entities, "COURSE"),
        "instructors": unique_texts(ner_entities, "INSTRUCTOR"),
        "emails": unique_texts(regex_entities, "EMAIL"),
        "section_names": section_names_from_entities(ner_entities),
        "assignments": unique_texts(regex_entities, "ASSIGNMENT"),
        "readings": unique_texts(regex_entities, "READING"),
        "grading_weights": unique_texts(regex_entities, "GRADING_WEIGHT"),
        "due_dates": unique_texts(regex_entities, "DUE_DATE"),
        "course_dates": unique_texts(regex_entities, "COURSE_DATE"),
        "concepts": unique_texts(ner_entities, "CONCEPT"),
        "text_field": text_field,
        "entities": all_entities,
    }


def build_chat_example(record: dict, index: int, max_text_chars: int) -> dict:
    text_field, text = detect_text_field(record)
    trimmed_text = text[:max_text_chars]
    target = build_structured_target(record, index)
    user_prompt = (
        "Extract structured syllabus information from the following document.\n"
        "Return JSON with keys: "
        "document_id, source_url, course_codes, instructors, emails, section_names, "
        "assignments, readings, grading_weights, due_dates, course_dates, concepts, "
        "text_field, entities.\n\n"
        f"Document text:\n{trimmed_text}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=True)},
        ]
    }


def split_examples(examples: list[dict], validation_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(len(shuffled) * validation_ratio)
    if validation_ratio > 0 and validation_count == 0 and len(shuffled) > 1:
        validation_count = 1
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return train, validation


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def default_build_error_log(train_output: str | Path) -> Path:
    out = Path(train_output)
    return out.parent / f"{out.stem}_build_errors.jsonl"


def build_finetune_dataset(
    input_jsonl: str | Path,
    train_output: str | Path,
    valid_output: str | Path,
    *,
    validation_ratio: float = 0.1,
    max_text_chars: int = 12000,
    seed: int = 13,
    tolerant: bool = True,
    allow_empty_outputs: bool = False,
    error_log_path: str | Path | None = None,
) -> tuple[list[dict], list[dict]]:
    log_path: Path | None
    if error_log_path is not None:
        log_path = Path(error_log_path)
    elif tolerant:
        log_path = default_build_error_log(train_output)
    else:
        log_path = None

    if log_path and tolerant:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")

    rows = load_jsonl(
        input_jsonl,
        strict=not tolerant,
        error_log_path=log_path if tolerant else None,
        stage="build_finetune_load",
    )
    examples: list[dict] = []
    skipped = 0
    build_errors = 0
    for index, record in enumerate(rows, start=1):
        if not isinstance(record, dict):
            build_errors += 1
            if log_path and tolerant:
                log_pipeline_error(
                    log_path,
                    stage="build_finetune_row",
                    message="record is not a JSON object",
                    index=index,
                    path=str(input_jsonl),
                )
            continue
        if not has_usable_document_text(record):
            skipped += 1
            continue
        try:
            examples.append(build_chat_example(record, index, max_text_chars))
        except Exception as exc:
            build_errors += 1
            if not tolerant:
                raise
            record_id = normalize_record_id(record, index)
            print(f"[build] skip {record_id}: {exc}")
            if log_path:
                log_pipeline_error(
                    log_path,
                    stage="build_finetune_record",
                    message=str(exc),
                    exc_type=type(exc).__name__,
                    record_id=record_id,
                    index=index,
                    path=str(input_jsonl),
                )

    if not examples:
        write_jsonl(train_output, [])
        write_jsonl(valid_output, [])
        if not rows:
            return [], []
        if allow_empty_outputs:
            print(
                f"[build] warning: no SFT examples ({len(rows)} rows, {skipped} no text, {build_errors} errors). "
                f"Wrote empty {train_output} and {valid_output}."
            )
            return [], []
        raise ValueError(
            f"No usable document text in {input_jsonl!r} "
            f"({len(rows)} rows read, {skipped} skipped, {build_errors} build errors). "
            "Re-run ingest or remove rows with empty `text` / failed fetches. "
            "Use allow_empty_outputs=True to write empty train/valid files without raising."
        )

    train, valid = split_examples(examples, validation_ratio, seed)
    write_jsonl(train_output, train)
    write_jsonl(valid_output, valid)
    if build_errors or skipped:
        print(
            f"[build] train={len(train)} valid={len(valid)}; skipped_no_text={skipped}, build_errors={build_errors}"
        )
    return train, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/validation JSONL files for syllabus structured-output fine-tuning."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to ingested syllabus JSONL.")
    parser.add_argument(
        "--train-output",
        default="data/finetune/train.jsonl",
        help="Output path for train examples.",
    )
    parser.add_argument(
        "--valid-output",
        default="data/finetune/valid.jsonl",
        help="Output path for validation examples.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on invalid JSON lines or per-record build errors.",
    )
    parser.add_argument(
        "--allow-empty-outputs",
        action="store_true",
        help="If no examples remain, write empty train/valid JSONL instead of raising.",
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Error log path when tolerant (default: alongside train output *_build_errors.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, valid = build_finetune_dataset(
        args.input_jsonl,
        args.train_output,
        args.valid_output,
        validation_ratio=args.validation_ratio,
        max_text_chars=args.max_text_chars,
        seed=args.seed,
        tolerant=not args.strict,
        allow_empty_outputs=args.allow_empty_outputs,
        error_log_path=args.error_log,
    )
    print(f"Train examples: {len(train)}")
    print(f"Validation examples: {len(valid)}")
    print(f"Train output: {args.train_output}")
    print(f"Validation output: {args.valid_output}")


if __name__ == "__main__":
    main()
