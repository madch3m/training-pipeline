"""Tests against ``us_freshman_core_syllabi_index.csv`` in the repo root."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from syllabi_index_csv_to_jsonl import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_JSONL,
    csv_to_jsonl,
    iter_csv_records,
    row_to_record,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = REPO_ROOT / "us_freshman_core_syllabi_index.csv"


@pytest.mark.skipif(not INDEX_CSV.is_file(), reason="us_freshman_core_syllabi_index.csv not present")
def test_default_paths_point_at_repo_index():
    assert DEFAULT_INPUT_CSV.resolve() == INDEX_CSV.resolve()
    assert DEFAULT_OUTPUT_JSONL.name == "us_freshman_core_syllabi_urls.jsonl"


@pytest.mark.skipif(not INDEX_CSV.is_file(), reason="us_freshman_core_syllabi_index.csv not present")
def test_index_csv_has_expected_columns():
    with INDEX_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        names = set(reader.fieldnames)
    required = {
        "source_url",
        "domain",
        "title",
        "school",
        "department",
        "subject_area",
        "file_type",
        "local_file_path",
        "sha256_hash",
        "query_used",
        "download_timestamp",
        "validation_hits",
    }
    assert required.issubset(names)


@pytest.mark.skipif(not INDEX_CSV.is_file(), reason="us_freshman_core_syllabi_index.csv not present")
def test_index_csv_row_count_and_urls():
    rows = list(iter_csv_records(INDEX_CSV))
    assert len(rows) == 100
    for row in rows:
        url = row["source_url"].strip()
        assert url.startswith("https://"), url


@pytest.mark.skipif(not INDEX_CSV.is_file(), reason="us_freshman_core_syllabi_index.csv not present")
def test_csv_to_jsonl_roundtrip_sample(tmp_path):
    out = tmp_path / "urls.jsonl"
    count = csv_to_jsonl(INDEX_CSV, out, max_rows=5)
    assert count == 5
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["source_url"].startswith("https://")
    assert first["id"] == first["sha256_hash"]
    assert first["school"]


@pytest.mark.skipif(not INDEX_CSV.is_file(), reason="us_freshman_core_syllabi_index.csv not present")
def test_csv_to_jsonl_full_index(tmp_path):
    out = tmp_path / "all.jsonl"
    count = csv_to_jsonl(INDEX_CSV, out)
    assert count == 100
    assert len(out.read_text(encoding="utf-8").splitlines()) == 100


def test_row_to_record_sets_id_from_hash():
    row = {
        "source_url": "https://example.edu/s.pdf",
        "sha256_hash": "abc" * 10 + "def",
        "school": "Test U",
    }
    rec = row_to_record(row, 1)
    assert rec["id"] == row["sha256_hash"]


def test_iter_csv_records_requires_source_url_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_url"):
        list(iter_csv_records(bad))
