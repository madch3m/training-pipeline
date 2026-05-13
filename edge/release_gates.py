"""Evaluate go/no-go deployment gates from edge benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check edge release gates from benchmark/eval reports.")
    p.add_argument("--benchmark-json", default="reports/edge_benchmark.json")
    p.add_argument("--min-parse-success-rate", type=float, default=0.98)
    p.add_argument("--min-study-plan-f1", type=float, default=0.35)
    p.add_argument("--max-p95-latency-seconds", type=float, default=3.5)
    p.add_argument("--max-peak-ram-bytes", type=int, default=1_500_000_000)
    p.add_argument("--output-json", default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def evaluate_gates(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    quality = report.get("quality", {})
    latency = report.get("latency_sec", {})
    memory = report.get("memory_bytes", {})

    parse_rate = float(quality.get("json_parse_success_rate", 0.0))
    study_f1 = float(quality.get("mean_study_plan_macro_f1", 0.0))
    p95 = float(latency.get("p95", 9999.0))
    peak = int(memory.get("peak", 10**18))

    gates = {
        "parse_success_rate_ok": parse_rate >= args.min_parse_success_rate,
        "study_plan_f1_ok": study_f1 >= args.min_study_plan_f1,
        "p95_latency_ok": p95 <= args.max_p95_latency_seconds,
        "peak_ram_ok": peak <= args.max_peak_ram_bytes,
    }
    violations = [k for k, ok in gates.items() if not ok]
    return {
        "ok": not violations,
        "gates": gates,
        "violations": violations,
        "metrics": {
            "json_parse_success_rate": parse_rate,
            "mean_study_plan_macro_f1": study_f1,
            "p95_latency_seconds": p95,
            "peak_ram_bytes": peak,
        },
        "thresholds": {
            "min_parse_success_rate": args.min_parse_success_rate,
            "min_study_plan_f1": args.min_study_plan_f1,
            "max_p95_latency_seconds": args.max_p95_latency_seconds,
            "max_peak_ram_bytes": args.max_peak_ram_bytes,
        },
    }


def main() -> None:
    args = parse_args()
    bench_path = Path(args.benchmark_json)
    if not bench_path.exists():
        raise SystemExit(f"Benchmark report missing: {bench_path}")

    report = json.loads(bench_path.read_text(encoding="utf-8"))
    result = evaluate_gates(report, args)
    text = json.dumps(result, indent=2, ensure_ascii=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.strict and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
