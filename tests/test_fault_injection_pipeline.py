"""Fault-injection style checks for tolerant CSV → process path."""

import json
from pathlib import Path

from build_finetune_dataset import build_finetune_dataset
from process_syllabi_jsonl import process_jsonl
from syllabi_index_csv_to_jsonl import csv_to_jsonl


def test_csv_process_survives_malformed_extra_row(tmp_path: Path):
    csv_path = tmp_path / "index.csv"
    csv_path.write_text(
        "source_url,domain,id\n"
        "https://good.edu/1,good.edu,a1\n"
        ",bad.edu,skipme\n",
        encoding="utf-8",
    )
    url_jsonl = tmp_path / "urls.jsonl"
    n = csv_to_jsonl(csv_path, url_jsonl)
    assert n == 1
    text_jsonl = tmp_path / "text.jsonl"
    text_jsonl.write_text(
        json.dumps({"id": "a1", "text": "CS 101 syllabus\nGrading: Homework 40%\n" * 2}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "ent.jsonl"
    rows = process_jsonl(text_jsonl, out)
    assert len(rows) == 1


def test_build_tolerates_bad_json_line(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    good = {"id": "g", "text": "CS 101 syllabus\nGrading: Homework 40%\nInstructor: Jane Smith"}
    inp.write_text(json.dumps(good) + "\nBOGUS\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    tr, va = build_finetune_dataset(inp, train, valid, validation_ratio=0.0, seed=0)
    assert len(tr) == 1 and len(va) == 0
