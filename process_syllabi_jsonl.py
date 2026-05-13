from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pipeline_errors import log_pipeline_error


TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "body",
    "document_text",
    "cleaned_text",
    "html_text",
    "pdf_text",
    "raw_text",
)

SECTION_HEADERS = (
    "required readings",
    "required reading",
    "textbook",
    "textbooks",
    "course schedule",
    "weekly schedule",
    "assignments",
    "grading",
    "office hours",
    "learning objectives",
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s?-?\s?\d{2,3}[A-Z]?\b")
PERCENT_RE = re.compile(r"\b\d{1,3}%")
DATE_RE = re.compile(
    r"\b(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?"
    r"|"
    r"\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})"
    r")\b",
    re.IGNORECASE,
)
ASSIGNMENT_LINE_RE = re.compile(
    r"^(?:assignment|project|paper|exam|quiz|midterm|final)\b.*$",
    re.IGNORECASE,
)
READING_LINE_RE = re.compile(
    r"^(?:required reading|required readings|reading|textbook|readings)\b.*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Span:
    label: str
    text: str
    start: int
    end: int
    source: str


def load_jsonl(
    path: str | Path,
    *,
    strict: bool = True,
    error_log_path: str | Path | None = None,
    stage: str = "load_jsonl",
) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"expected object, got {type(row).__name__}")
            rows.append(row)
        except (json.JSONDecodeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if error_log_path:
                log_pipeline_error(
                    error_log_path,
                    stage=stage,
                    message=str(exc),
                    line=line_number,
                    path=str(path),
                    snippet=stripped[:240],
                )
    return rows


def detect_text_field(record: dict) -> tuple[str, str]:
    for field in TEXT_FIELD_CANDIDATES:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return field, value
    for key, value in record.items():
        if isinstance(value, str) and len(value.split()) >= 30:
            return key, value
    raise ValueError("Could not determine a text field for record")


def has_usable_document_text(record: dict) -> bool:
    """True if ``detect_text_field`` succeeds and the chosen string has non-whitespace content."""
    try:
        _, text = detect_text_field(record)
    except ValueError:
        return False
    return bool(text.strip())


def normalize_record_id(record: dict, index: int) -> str:
    value = record.get("id") or record.get("doc_id") or record.get("uuid")
    if value:
        return str(value)
    return f"record_{index:04d}"


def extract_regex_entities(text: str) -> list[Span]:
    entities: list[Span] = []

    for match in EMAIL_RE.finditer(text):
        entities.append(Span("EMAIL", match.group(0), match.start(), match.end(), "regex"))

    for match in COURSE_CODE_RE.finditer(text):
        entities.append(Span("COURSE", match.group(0), match.start(), match.end(), "regex"))

    for match in PERCENT_RE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line_text = text[line_start:line_end].lower()
        left = text[max(0, match.start() - 60):match.start()].lower()
        label = "GRADING_WEIGHT" if any(
            token in (line_text + " " + left)
            for token in ("grading", "grade", "homework", "exam", "project", "quiz", "participation")
        ) else "PERCENT"
        entities.append(Span(label, match.group(0), match.start(), match.end(), "regex"))

    for match in DATE_RE.finditer(text):
        window = text[max(0, match.start() - 50):min(len(text), match.end() + 50)].lower()
        label = "DUE_DATE" if any(token in window for token in ("due", "submit", "submission")) else "COURSE_DATE"
        entities.append(Span(label, match.group(0), match.start(), match.end(), "regex"))

    offset = 0
    for line in text.splitlines():
        stripped = line.strip()
        if READING_LINE_RE.match(stripped):
            start = offset + line.find(stripped)
            entities.append(Span("READING", stripped, start, start + len(stripped), "regex"))
        elif ASSIGNMENT_LINE_RE.match(stripped):
            start = offset + line.find(stripped)
            entities.append(Span("ASSIGNMENT", stripped, start, start + len(stripped), "regex"))
        offset += len(line) + 1

    return dedupe_spans(entities)


def extract_heuristic_ner(text: str) -> list[Span]:
    entities: list[Span] = []
    lowered = text.lower()

    for header in SECTION_HEADERS:
        start = lowered.find(header)
        while start != -1:
            entities.append(Span("SECTION", text[start:start + len(header)], start, start + len(header), "heuristic_ner"))
            start = lowered.find(header, start + len(header))

    instructor_patterns = (
        re.compile(r"\bprof(?:essor)?\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
        re.compile(r"\binstructor:\s*[A-Z][^\n,;]+", re.IGNORECASE),
    )
    for pattern in instructor_patterns:
        for match in pattern.finditer(text):
            entities.append(Span("INSTRUCTOR", match.group(0).strip(), match.start(), match.end(), "heuristic_ner"))

    concept_pattern = re.compile(
        r"\b(?:topic|topics|concept|concepts|unit|week)\s*[:\-]\s*([^\n]{3,120})",
        re.IGNORECASE,
    )
    for match in concept_pattern.finditer(text):
        span_text = match.group(1).strip()
        start = match.start(1)
        entities.append(Span("CONCEPT", span_text, start, start + len(span_text), "heuristic_ner"))

    return dedupe_spans(entities)


def dedupe_spans(spans: Iterable[Span]) -> list[Span]:
    seen: set[tuple[str, int, int, str]] = set()
    deduped: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label, item.source)):
        key = (span.label, span.start, span.end, span.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def build_output_record(record: dict, index: int) -> dict:
    text_field, text = detect_text_field(record)
    regex_entities = [asdict(item) for item in extract_regex_entities(text)]
    ner_entities = [asdict(item) for item in extract_heuristic_ner(text)]

    return {
        "id": normalize_record_id(record, index),
        "source_url": record.get("source_url"),
        "text_field": text_field,
        "regex_entities": regex_entities,
        "ner_entities": ner_entities,
        "input_record": record,
    }


def default_process_error_log(output_path: str | Path) -> Path:
    out = Path(output_path)
    return out.parent / f"{out.stem}_errors.jsonl"


def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tolerant: bool = True,
    strict_jsonl: bool | None = None,
    error_log_path: str | Path | None = None,
) -> list[dict]:
    """Run labeling. If ``tolerant`` (default), skip bad JSON lines and per-record failures; log to *error_log_path*."""
    if strict_jsonl is not None:
        tolerant = not strict_jsonl
    log_path: Path | None
    if error_log_path is not None:
        log_path = Path(error_log_path)
    elif tolerant:
        log_path = default_process_error_log(output_path)
    else:
        log_path = None

    if log_path and tolerant:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")

    rows = load_jsonl(
        input_path,
        strict=not tolerant,
        error_log_path=log_path if tolerant else None,
        stage="process_jsonl_load",
    )
    processed: list[dict] = []
    skipped_no_text = 0
    skipped_record_error = 0
    for index, record in enumerate(rows, start=1):
        if not isinstance(record, dict):
            skipped_record_error += 1
            if log_path and tolerant:
                log_pipeline_error(
                    log_path,
                    stage="process_jsonl_row",
                    message="record is not a JSON object",
                    index=index,
                    path=str(input_path),
                )
            continue
        if not has_usable_document_text(record):
            skipped_no_text += 1
            record_id = normalize_record_id(record, index)
            print(f"[process] skip {record_id}: no usable document text")
            continue
        try:
            processed.append(build_output_record(record, index))
        except Exception as exc:
            skipped_record_error += 1
            if not tolerant:
                raise
            record_id = normalize_record_id(record, index)
            print(f"[process] skip {record_id}: {exc}")
            if log_path:
                log_pipeline_error(
                    log_path,
                    stage="process_jsonl_record",
                    message=str(exc),
                    exc_type=type(exc).__name__,
                    record_id=record_id,
                    index=index,
                    path=str(input_path),
                )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not processed:
        output.write_text("", encoding="utf-8")
        if not rows:
            return []
        if tolerant:
            print(
                f"[process] warning: no labeled rows ({len(rows)} loaded, "
                f"{skipped_no_text} no text, {skipped_record_error} errors). "
                f"Output empty: {output_path}"
            )
            return []
        hint = ""
        if skipped_no_text == len(rows):
            hint = " All rows lacked non-empty text (check ingest: empty PDF/HTML extractions or ingest_error rows)."
        raise ValueError(
            f"No usable document text in {input_path!r} "
            f"({len(rows)} rows read, {skipped_no_text} skipped).{hint}"
        )

    with output.open("w", encoding="utf-8") as handle:
        for item in processed:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    if skipped_record_error or skipped_no_text:
        print(
            f"[process] wrote {len(processed)} rows; skipped_no_text={skipped_no_text}, record_errors={skipped_record_error}"
        )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regex extraction and heuristic NER over syllabus JSONL input."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to the source JSONL file.")
    parser.add_argument(
        "--output-jsonl",
        default="data/labeled/syllabus_entities.jsonl",
        help="Path for the enriched JSONL output.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on invalid JSONL lines or per-record errors (default: tolerant).",
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Append/load error details here when tolerant (default: next to output, *_errors.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = process_jsonl(
        args.input_jsonl,
        args.output_jsonl,
        tolerant=not args.strict,
        error_log_path=args.error_log,
    )
    print(f"Processed {len(processed)} records (with usable text).")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
