from edge.privacy_telemetry import aggregate
from edge.release_gates import evaluate_gates


class _Args:
    min_parse_success_rate = 0.98
    min_study_plan_f1 = 0.30
    max_p95_latency_seconds = 3.5
    max_peak_ram_bytes = 1_500_000_000


def test_release_gates_pass():
    report = {
        "quality": {"json_parse_success_rate": 0.99, "mean_study_plan_macro_f1": 0.42},
        "latency_sec": {"p95": 2.8},
        "memory_bytes": {"peak": 1_000_000_000},
    }
    out = evaluate_gates(report, _Args())
    assert out["ok"] is True
    assert out["violations"] == []


def test_privacy_telemetry_aggregate_counts():
    benchmark = {
        "per_example": [
            {"latency_sec": 0.9, "output_tokens": 120, "parsed": True, "study_plan_macro_f1": 0.7},
            {"latency_sec": 2.4, "output_tokens": 620, "parsed": False, "study_plan_macro_f1": 0.2},
        ]
    }
    agg = aggregate(benchmark)
    assert agg["examples"] == 2
    assert agg["parse_counts"]["parsed_ok"] == 1
    assert agg["parse_counts"]["parsed_fail"] == 1
