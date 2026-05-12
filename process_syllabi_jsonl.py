from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
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


def process_jsonl(input_path: str | Path, output_path: str | Path) -> list[dict]:
    rows = load_jsonl(input_path)
    processed = [build_output_record(record, index) for index, record in enumerate(rows, start=1)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in processed:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = process_jsonl(args.input_jsonl, args.output_jsonl)
    print(f"Processed {len(processed)} records.")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
