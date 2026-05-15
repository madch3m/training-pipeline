"""Label syllabi with GPT-4o using strict JSON-Schema mode.

Three modes:
- ``all``       : label every doc.
- ``golden``    : label only docs whose document_id is in --golden-ids file.
                  Use this on the same set you hand-label, to get a teacher-ceiling.
- ``arbitrate`` : label only docs whose document_id is in --arbitration-ids file.
                  Use this on docs where DeepSeek output looks low-confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from label_with_deepseek import SYSTEM_PROMPT, build_user_prompt
from process_syllabi_jsonl import (
    detect_text_field,
    extract_heuristic_ner,
    extract_regex_entities,
    has_usable_document_text,
    load_jsonl,
    normalize_record_id,
)
from syllabus_schema import SyllabusFacts


def to_openai_strict_schema(model_cls) -> dict[str, Any]:
    """Pydantic JSON Schema → OpenAI strict-mode schema.

    OpenAI strict mode requires every object to set additionalProperties=false and
    list every property in ``required`` (nullables use anyOf-with-null, which Pydantic
    already emits for ``Optional`` fields).
    """

    raw = model_cls.model_json_schema()

    def harden(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for v in node.values():
                harden(v)
        elif isinstance(node, list):
            for item in node:
                harden(item)

    harden(raw)
    return raw


SCHEMA = to_openai_strict_schema(SyllabusFacts)


def call_openai(client, model: str, system_prompt: str, user_prompt: str, max_retries: int = 3) -> tuple[str, int, int]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "SyllabusFacts", "schema": SCHEMA, "strict": True},
                },
                temperature=0.0,
            )
            return (
                response.choices[0].message.content or "",
                int(response.usage.prompt_tokens),
                int(response.usage.completion_tokens),
            )
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gpt-4o call failed after {max_retries} attempts: {last_err}")


def _load_id_set(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label syllabi with GPT-4o (strict JSON schema).")
    p.add_argument("--input-jsonl", default="data/ingested/us_freshman_core_syllabi_with_text.jsonl")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-text-chars", type=int, default=20_000)
    p.add_argument(
        "--mode",
        choices=["all", "golden", "arbitrate"],
        default="all",
    )
    p.add_argument("--golden-ids", default=None, help="One document_id per line; required for --mode golden.")
    p.add_argument("--arbitration-ids", default=None, help="One document_id per line; required for --mode arbitrate.")
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set ${args.api_key_env} before running.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    filter_ids: set[str] | None = None
    if args.mode == "golden":
        filter_ids = _load_id_set(args.golden_ids)
        if filter_ids is None:
            raise SystemExit("--mode golden requires --golden-ids")
    elif args.mode == "arbitrate":
        filter_ids = _load_id_set(args.arbitration_ids)
        if filter_ids is None:
            raise SystemExit("--mode arbitrate requires --arbitration-ids")

    rows = load_jsonl(args.input_jsonl, strict=False)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_skip = 0
    total_in = total_out = 0
    with output.open("w", encoding="utf-8") as fh:
        for i, record in enumerate(rows, 1):
            if args.max_rows and (n_ok + n_fail) >= args.max_rows:
                break
            if not isinstance(record, dict) or not has_usable_document_text(record):
                n_skip += 1
                continue
            doc_id = normalize_record_id(record, i)
            if filter_ids is not None and doc_id not in filter_ids:
                n_skip += 1
                continue

            _, text = detect_text_field(record)
            text = text[: args.max_text_chars]
            regex_entities = [e.__dict__ for e in extract_regex_entities(text)]
            ner_entities = [e.__dict__ for e in extract_heuristic_ner(text)]
            sections = sorted({e["text"] for e in ner_entities if e["label"] == "SECTION"})
            user_prompt = build_user_prompt(
                text, regex_entities + ner_entities, sections, doc_id, record.get("source_url")
            )

            raw = ""
            try:
                raw, in_tok, out_tok = call_openai(client, args.model, SYSTEM_PROMPT, user_prompt)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("non_object_json")
                payload.setdefault("document_id", doc_id)
                if record.get("source_url"):
                    payload.setdefault("source_url", record.get("source_url"))
                facts = SyllabusFacts.model_validate(payload)
                fh.write(
                    json.dumps(
                        {"facts": facts.model_dump(mode="json"), "tokens": {"in": in_tok, "out": out_tok}},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                n_ok += 1
                total_in += in_tok
                total_out += out_tok
            except Exception as exc:
                fh.write(
                    json.dumps(
                        {"document_id": doc_id, "error": str(exc), "raw": raw[:2000]},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                n_fail += 1
            fh.flush()
            if i % 10 == 0:
                print(f"[gpt-4o] {i} processed: ok={n_ok} fail={n_fail} skip={n_skip}")

    # Approximate per-million-token pricing for cost estimation. Falls back to a
    # generic ratio when the model isn't in the table; verify exact spend in the
    # OpenAI billing dashboard.
    pricing_usd_per_m: dict[str, tuple[float, float]] = {
        "gpt-5": (10.00, 30.00),
        "gpt-5-mini": (1.00, 4.00),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-2024-11-20": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }
    in_rate, out_rate = pricing_usd_per_m.get(args.model, (5.00, 15.00))
    cost = (total_in / 1e6 * in_rate) + (total_out / 1e6 * out_rate)
    print(
        f"[{args.model}] DONE ok={n_ok} fail={n_fail} skip={n_skip} | "
        f"tokens in={total_in} out={total_out} | est cost ~${cost:.2f} (approximate)"
    )


if __name__ == "__main__":
    main()
