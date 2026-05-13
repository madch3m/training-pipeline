"""Generate structured JSON with a trained LoRA adapter and compare to SFT labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluate_structured_json import (
    aggregate_compare_stats,
    assistant_payload_from_sft_example,
    compare_assistant_payloads,
    parse_assistant_json,
)


def _read_base_model_from_adapter(adapter_path: Path) -> str | None:
    cfg = adapter_path / "adapter_config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    name = data.get("base_model_name_or_path")
    return str(name) if name else None


def _strip_generation_fences(text: str) -> str:
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return s


def prompt_messages_from_sft_example(example: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    out: list[dict[str, str]] = []
    for msg in messages[:-1]:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        out.append({"role": str(role), "content": content})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trained syllabus extractor on SFT JSONL and score vs labels.")
    p.add_argument(
        "--adapter-path",
        default="artifacts/hf_syllabus_extractor",
        help="Directory with LoRA adapter + tokenizer (from train_hf_structured_extractor).",
    )
    p.add_argument(
        "--base-model-name",
        default=None,
        help="Override base model id (default: read from adapter_config.json).",
    )
    p.add_argument(
        "--sft-jsonl",
        default="data/finetune/valid.jsonl",
        help="Chat JSONL (messages with final assistant JSON).",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Limit evaluated examples.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--device-map", default="auto", help="Transformers device_map (e.g. auto, cpu).")
    p.add_argument("--predictions-jsonl", default=None, help="Write per-row predictions + metrics here.")
    p.add_argument("--report-json", default=None, help="Write summary JSON here.")
    return p.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import syllabus_torch_compat

    syllabus_torch_compat.apply_torch_ao_compat_patches()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_path = Path(args.adapter_path).resolve()
    if not adapter_path.is_dir():
        return {"error": "adapter_path_not_found", "path": str(adapter_path)}

    base_name = args.base_model_name or _read_base_model_from_adapter(adapter_path)
    if not base_name:
        return {
            "error": "missing_base_model",
            "hint": "Pass --base-model-name or ensure adapter_config.json exists under adapter path.",
        }

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    force_cpu = args.device_map == "cpu" or not use_cuda
    device_map = None if force_cpu else args.device_map
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    if force_cpu:
        model = model.to("cpu")

    model.eval()

    def _model_device() -> torch.device:
        try:
            return model.device  # type: ignore[attr-defined]
        except Exception:
            return next(model.parameters()).device

    sft_path = Path(args.sft_jsonl)
    if not sft_path.is_file():
        return {"error": "sft_jsonl_not_found", "path": str(sft_path.resolve())}

    per_example: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(sft_path.read_text(encoding="utf-8").splitlines(), start=1):
        if args.max_samples is not None and len(per_example) >= args.max_samples:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            example = json.loads(stripped)
        except json.JSONDecodeError as exc:
            row = {
                "line": line_no,
                "parsed": False,
                "parse_error": f"line_json:{exc}",
                "document_id_match": False,
                "macro_f1_string_lists": 0.0,
                "study_plan_macro_f1": 0.0,
            }
            per_example.append(row)
            prediction_rows.append({**row, "generated": "", "gold_raw": None})
            continue

        gold_raw, gerr = assistant_payload_from_sft_example(example)
        gold_obj, _ = parse_assistant_json(gold_raw) if gold_raw else (None, None)

        msgs = prompt_messages_from_sft_example(example)
        if not msgs:
            row = {
                "line": line_no,
                "parsed": False,
                "parse_error": "bad_messages",
                "document_id_match": False,
                "macro_f1_string_lists": 0.0,
                "study_plan_macro_f1": 0.0,
            }
            per_example.append(row)
            prediction_rows.append({**row, "generated": "", "gold_raw": gold_raw})
            continue

        input_ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(_model_device())

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            out = model.generate(input_ids, **gen_kwargs)
        gen_ids = out[0][input_ids.shape[-1] :]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned = _strip_generation_fences(generated)

        pred_obj, perr = parse_assistant_json(cleaned)
        parsed = pred_obj is not None and gold_obj is not None
        if parsed:
            metrics = compare_assistant_payloads(pred_obj, gold_obj)
            row = {
                "line": line_no,
                "parsed": True,
                "parse_error": None,
                "document_id_match": metrics["document_id_match"],
                "macro_f1_string_lists": metrics["macro_f1_string_lists"],
                "field_f1": metrics["field_f1"],
                "study_plan_macro_f1": metrics["study_plan_macro_f1"],
                "study_plan_field_f1": metrics["study_plan_field_f1"],
                "entities_length_match": metrics["entities_length_match"],
                "text_field_match": metrics["text_field_match"],
            }
        else:
            err = perr or (f"gold_parse_failed:{gerr}" if gold_obj is None and gerr else "pred_or_gold_parse_failed")
            row = {
                "line": line_no,
                "parsed": False,
                "parse_error": err,
                "document_id_match": False,
                "macro_f1_string_lists": 0.0,
                "study_plan_macro_f1": 0.0,
            }
        per_example.append(row)
        prediction_rows.append(
            {
                **row,
                "generated": generated,
                "gold_raw": gold_raw,
            }
        )

    summary = aggregate_compare_stats(per_example)
    summary.update(
        {
            "adapter_path": str(adapter_path),
            "base_model_name": base_name,
            "sft_jsonl": str(sft_path.resolve()),
            "examples_evaluated": len(per_example),
        }
    )
    return {"summary": summary, "per_example": per_example, "prediction_rows": prediction_rows}


def main() -> None:
    args = parse_args()
    result = run_evaluation(args)
    if "error" in result:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    summary = result["summary"]
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    print(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(
            json.dumps({"summary": summary, "per_example": result["per_example"]}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if args.predictions_jsonl:
        out_p = Path(args.predictions_jsonl)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            for row in result["prediction_rows"]:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
