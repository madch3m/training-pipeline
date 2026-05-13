"""Emit a machine-readable data report for ML evaluation (counts, domains, overlap)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_structured_json import evaluate_sft_jsonl
from pipeline_runner import SyllabusPipelineConfig, find_repo_root


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def domain_histogram_from_csv(csv_path: Path, limit: int = 25) -> dict[str, Any]:
    if not csv_path.exists():
        return {"error": "csv_missing", "path": str(csv_path)}
    counts: Counter[str] = Counter()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            d = (row.get("domain") or "").strip()
            if d:
                counts[d] += 1
    top = counts.most_common(limit)
    return {
        "unique_domains": len(counts),
        "top_domains": [{"domain": d, "count": c} for d, c in top],
        "total_rows": sum(counts.values()),
    }


def document_ids_from_sft(path: Path) -> list[str]:
    ids: list[str] = []
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ex = json.loads(s)
        except json.JSONDecodeError:
            continue
        msgs = ex.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        last = msgs[-1]
        if not isinstance(last, dict) or last.get("content") is None:
            continue
        try:
            payload = json.loads(last["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("document_id"):
            ids.append(str(payload["document_id"]))
    return ids


def train_valid_overlap(train_path: Path, valid_path: Path) -> dict[str, Any]:
    train_ids = set(document_ids_from_sft(train_path))
    valid_ids = set(document_ids_from_sft(valid_path))
    overlap = train_ids & valid_ids
    return {
        "train_document_ids": len(train_ids),
        "valid_document_ids": len(valid_ids),
        "overlap_count": len(overlap),
        "overlap_examples": sorted(overlap)[:10],
    }


def file_sha256(path: Path, chunk: int = 65536) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_report(cfg: SyllabusPipelineConfig) -> dict[str, Any]:
    rr = cfg.repo_root
    rep: dict[str, Any] = {
        "repo_root": str(rr.resolve()),
        "artifact_hashes": {
            "index_csv_sha256": file_sha256(cfg.index_csv),
        },
        "line_counts": {
            "url_jsonl": count_nonempty_lines(cfg.url_jsonl),
            "text_jsonl": count_nonempty_lines(cfg.text_jsonl),
            "entities_jsonl": count_nonempty_lines(cfg.entities_jsonl),
            "train_jsonl": count_nonempty_lines(cfg.train_jsonl),
            "valid_jsonl": count_nonempty_lines(cfg.valid_jsonl),
        },
        "domain_histogram": domain_histogram_from_csv(cfg.index_csv),
        "train_valid_ids": train_valid_overlap(cfg.train_jsonl, cfg.valid_jsonl),
    }
    if cfg.train_jsonl.exists():
        rep["train_sft_json_eval"] = evaluate_sft_jsonl(cfg.train_jsonl)
    if cfg.valid_jsonl.exists():
        rep["valid_sft_json_eval"] = evaluate_sft_jsonl(cfg.valid_jsonl)
    return rep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ML evaluation data report for the syllabus pipeline.")
    p.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: discover via build_finetune_dataset.py).",
    )
    p.add_argument("--output-json", default=None, help="Write report JSON to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    cfg = SyllabusPipelineConfig.for_repo(root)
    report = build_report(cfg)
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
