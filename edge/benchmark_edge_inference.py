"""Benchmark quantized or adapter-based edge inference for latency, memory, and quality."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

from evaluate_structured_json import (
    aggregate_compare_stats,
    assistant_payload_from_sft_example,
    compare_assistant_payloads,
    parse_assistant_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run edge-oriented inference benchmark.")
    p.add_argument("--adapter-path", default="artifacts/hf_syllabus_extractor")
    p.add_argument("--sft-jsonl", default="data/finetune/valid.jsonl")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--device-map", default="cpu")
    p.add_argument("--report-json", default="reports/edge_benchmark.json")
    return p.parse_args()


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def prompt_messages(example: dict[str, Any]) -> list[dict[str, str]] | None:
    msgs = example.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    out: list[dict[str, str]] = []
    for m in msgs[:-1]:
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    return out


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import syllabus_torch_compat

    syllabus_torch_compat.apply_torch_ao_compat_patches()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_path = Path(args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available() and args.device_map != "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        device_map=None if not use_cuda else args.device_map,
        torch_dtype=dtype,
    )
    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    def model_device() -> torch.device:
        try:
            return model.device  # type: ignore[attr-defined]
        except Exception:
            return next(model.parameters()).device

    per_example: list[dict[str, Any]] = []
    latencies: list[float] = []
    output_tokens: list[int] = []
    tracemalloc.start()

    lines = Path(args.sft_jsonl).read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if len(per_example) >= args.max_samples:
            break
        s = line.strip()
        if not s:
            continue
        try:
            ex = json.loads(s)
        except json.JSONDecodeError:
            continue
        prompts = prompt_messages(ex)
        gold_raw, _ = assistant_payload_from_sft_example(ex)
        gold_obj, _ = parse_assistant_json(gold_raw) if gold_raw else (None, None)
        if not prompts or gold_obj is None:
            continue

        input_ids = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model_device())

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
        latencies.append(dt)

        gen_ids = out[0][input_ids.shape[-1] :]
        output_tokens.append(int(gen_ids.shape[-1]))
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_obj, perr = parse_assistant_json(generated)
        if pred_obj is None:
            row = {
                "line": line_no,
                "parsed": False,
                "parse_error": perr,
                "latency_sec": dt,
                "output_tokens": int(gen_ids.shape[-1]),
                "macro_f1_string_lists": 0.0,
                "study_plan_macro_f1": 0.0,
                "document_id_match": False,
            }
        else:
            m = compare_assistant_payloads(pred_obj, gold_obj)
            row = {
                "line": line_no,
                "parsed": True,
                "parse_error": None,
                "latency_sec": dt,
                "output_tokens": int(gen_ids.shape[-1]),
                "macro_f1_string_lists": m["macro_f1_string_lists"],
                "study_plan_macro_f1": m["study_plan_macro_f1"],
                "document_id_match": m["document_id_match"],
            }
        per_example.append(row)

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    quality_summary = aggregate_compare_stats(per_example)
    report = {
        "adapter_path": str(adapter_path.resolve()),
        "sft_jsonl": str(Path(args.sft_jsonl).resolve()),
        "examples": len(per_example),
        "latency_sec": {
            "p50": _p(latencies, 0.50),
            "p95": _p(latencies, 0.95),
            "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "output_tokens": {
            "p50": _p([float(x) for x in output_tokens], 0.50),
            "p95": _p([float(x) for x in output_tokens], 0.95),
            "mean": statistics.mean(output_tokens) if output_tokens else 0.0,
        },
        "memory_bytes": {"current": cur_mem, "peak": peak_mem},
        "quality": quality_summary,
        "per_example": per_example,
    }
    return report


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
