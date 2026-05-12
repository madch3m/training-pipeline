"""Convert a syllabus index CSV (e.g. ``us_freshman_core_syllabi_index.csv``) to JSONL for ``ingest_jsonl_from_urls``.

Each CSV row becomes one JSON object per line. ``source_url`` is required so the
ingest step can fetch documents. If ``id`` is missing, ``sha256_hash`` (when
present) is copied to ``id`` for stable downstream keys.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = _PACKAGE_DIR / "us_freshman_core_syllabi_index.csv"
DEFAULT_OUTPUT_JSONL = _PACKAGE_DIR / "data" / "ingested" / "us_freshman_core_syllabi_urls.jsonl"


def iter_csv_records(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        if "source_url" not in reader.fieldnames:
            raise ValueError("CSV must include a source_url column.")
        yield from reader


def row_to_record(row: dict[str, str], index: int) -> dict[str, str]:
    record = {key: (value or "").strip() for key, value in row.items() if key}
    if not record.get("source_url"):
        raise ValueError(f"Row {index}: empty source_url")
    if not record.get("id") and record.get("sha256_hash"):
        record["id"] = record["sha256_hash"]
    return record


def csv_to_jsonl(
    input_csv: str | Path,
    output_jsonl: str | Path,
    *,
    max_rows: int | None = None,
    skip_empty_url: bool = True,
) -> int:
    written = 0
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_csv_records(input_csv), start=1):
            if max_rows is not None and written >= max_rows:
                break
            try:
                if skip_empty_url and not (row.get("source_url") or "").strip():
                    continue
                record = row_to_record(row, index)
            except ValueError:
                continue
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert syllabus index CSV to URL JSONL for ingest_jsonl_from_urls.")
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help=f"Path to index CSV (default: {DEFAULT_INPUT_CSV.name} next to this script).",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help=f"Path for JSONL output (default: {DEFAULT_OUTPUT_JSONL.relative_to(_PACKAGE_DIR)}).",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Write at most this many rows (for smoke tests).")
    parser.add_argument(
        "--keep-empty-url-rows",
        action="store_true",
        help="Do not skip rows with blank source_url (default: skip).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = csv_to_jsonl(
        args.input_csv,
        args.output_jsonl,
        max_rows=args.max_rows,
        skip_empty_url=not args.keep_empty_url_rows,
    )
    print(f"Wrote {count} JSONL records.")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
