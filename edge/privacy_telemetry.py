"""Aggregate privacy-safe telemetry counters from per-example benchmark rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def latency_bucket(seconds: float) -> str:
    if seconds < 1.0:
        return "<1s"
    if seconds < 2.0:
        return "1-2s"
    if seconds < 3.5:
        return "2-3.5s"
    return ">3.5s"


def token_bucket(tokens: int) -> str:
    if tokens < 256:
        return "<256"
    if tokens < 512:
        return "256-511"
    if tokens < 768:
        return "512-767"
    return ">=768"


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("per_example")
    if not isinstance(rows, list):
        rows = []

    latency_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    parse_counts: Counter[str] = Counter()
    study_f1_counts: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        latency = float(row.get("latency_sec", 0.0))
        tokens = int(row.get("output_tokens", 0))
        parsed = bool(row.get("parsed"))
        study_f1 = float(row.get("study_plan_macro_f1", 0.0))
        latency_counts[latency_bucket(latency)] += 1
        token_counts[token_bucket(tokens)] += 1
        parse_counts["parsed_ok" if parsed else "parsed_fail"] += 1
        study_f1_counts[
            "study_f1_high" if study_f1 >= 0.6 else ("study_f1_mid" if study_f1 >= 0.3 else "study_f1_low")
        ] += 1

    return {
        "examples": sum(parse_counts.values()),
        "parse_counts": dict(parse_counts),
        "latency_buckets": dict(latency_counts),
        "output_token_buckets": dict(token_counts),
        "study_plan_f1_buckets": dict(study_f1_counts),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build privacy-safe aggregate telemetry.")
    p.add_argument("--benchmark-json", default="reports/edge_benchmark.json")
    p.add_argument("--output-json", default="reports/edge_telemetry_aggregate.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    out = aggregate(report)
    text = json.dumps(out, indent=2, ensure_ascii=True)
    print(text)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
