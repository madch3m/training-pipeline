"""Pre-training quality gates for SFT data and split integrity."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from evaluate_structured_json import evaluate_sft_jsonl


def _assistant_payload_from_line(line: str) -> dict[str, Any] | None:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, dict):
        return None
    content = last.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _domains_from_sft(path: Path) -> set[str]:
    domains: set[str] = set()
    if not path.exists():
        return domains
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        payload = _assistant_payload_from_line(s)
        if not payload:
            continue
        source_url = payload.get("source_url")
        if not isinstance(source_url, str) or not source_url.strip():
            continue
        try:
            parsed = urlparse(source_url)
        except ValueError:
            continue
        host = (parsed.netloc or "").strip().lower()
        if host:
            domains.add(host)
    return domains


def _document_ids_from_sft(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        payload = _assistant_payload_from_line(s)
        if not payload:
            continue
        doc_id = payload.get("document_id")
        if doc_id is not None:
            ids.add(str(doc_id))
    return ids


def _train_valid_overlap(train_path: Path, valid_path: Path) -> dict[str, Any]:
    train_ids = _document_ids_from_sft(train_path)
    valid_ids = _document_ids_from_sft(valid_path)
    overlap = train_ids & valid_ids
    return {
        "train_document_ids": len(train_ids),
        "valid_document_ids": len(valid_ids),
        "overlap_count": len(overlap),
        "overlap_examples": sorted(overlap)[:10],
    }


@dataclass
class ReadinessThresholds:
    min_parse_rate: float = 0.995
    min_schema_clean_rate: float = 0.995
    max_document_id_overlap: int = 0
    max_domain_overlap_ratio: float = 0.70


def evaluate_readiness(
    train_jsonl: Path,
    valid_jsonl: Path,
    thresholds: ReadinessThresholds,
) -> dict[str, Any]:
    train_eval = evaluate_sft_jsonl(train_jsonl)
    valid_eval = evaluate_sft_jsonl(valid_jsonl)
    id_overlap = _train_valid_overlap(train_jsonl, valid_jsonl)

    train_domains = _domains_from_sft(train_jsonl)
    valid_domains = _domains_from_sft(valid_jsonl)
    overlap_domains = train_domains & valid_domains
    domain_overlap_ratio = (len(overlap_domains) / len(valid_domains)) if valid_domains else 0.0

    gates = {
        "parse_rate_train_ok": float(train_eval.get("parseable_rate", 0.0)) >= thresholds.min_parse_rate,
        "parse_rate_valid_ok": float(valid_eval.get("parseable_rate", 0.0)) >= thresholds.min_parse_rate,
        "schema_clean_train_ok": float(train_eval.get("schema_clean_rate", 0.0)) >= thresholds.min_schema_clean_rate,
        "schema_clean_valid_ok": float(valid_eval.get("schema_clean_rate", 0.0)) >= thresholds.min_schema_clean_rate,
        "doc_id_overlap_ok": int(id_overlap.get("overlap_count", 0)) <= thresholds.max_document_id_overlap,
        "domain_overlap_ok": domain_overlap_ratio <= thresholds.max_domain_overlap_ratio,
    }
    violations = [key for key, ok in gates.items() if not ok]

    return {
        "ok": not violations,
        "thresholds": asdict(thresholds),
        "gates": gates,
        "violations": violations,
        "train_eval": train_eval,
        "valid_eval": valid_eval,
        "document_id_overlap": id_overlap,
        "domain_split": {
            "train_domains": len(train_domains),
            "valid_domains": len(valid_domains),
            "overlap_domains": len(overlap_domains),
            "overlap_ratio_vs_valid": domain_overlap_ratio,
            "overlap_examples": sorted(overlap_domains)[:20],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check SFT JSON and split quality gates before training.")
    p.add_argument("--train-jsonl", default="data/finetune/train.jsonl")
    p.add_argument("--valid-jsonl", default="data/finetune/valid.jsonl")
    p.add_argument("--min-parse-rate", type=float, default=0.995)
    p.add_argument("--min-schema-clean-rate", type=float, default=0.995)
    p.add_argument("--max-document-id-overlap", type=int, default=0)
    p.add_argument("--max-domain-overlap-ratio", type=float, default=0.70)
    p.add_argument("--output-json", default=None)
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any gate fails.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = ReadinessThresholds(
        min_parse_rate=args.min_parse_rate,
        min_schema_clean_rate=args.min_schema_clean_rate,
        max_document_id_overlap=args.max_document_id_overlap,
        max_domain_overlap_ratio=args.max_domain_overlap_ratio,
    )
    report = evaluate_readiness(Path(args.train_jsonl), Path(args.valid_jsonl), thresholds)
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
