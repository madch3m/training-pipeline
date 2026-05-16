"""Bulk-label syllabus documents with DeepSeek-V3 → SyllabusFacts JSON.

Reads ingested text JSONL, runs the regex/heuristic pre-pass to attach in-context
hints, calls DeepSeek's OpenAI-compatible API, validates each response against the
``SyllabusFacts`` Pydantic schema, and writes one JSONL row per document.
Supports ``--resume`` to skip docs already present in the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from process_syllabi_jsonl import (
    detect_text_field,
    extract_heuristic_ner,
    extract_regex_entities,
    has_usable_document_text,
    load_jsonl,
    normalize_record_id,
)
from syllabus_schema import SyllabusFacts, to_strict_schema


# DeepSeek's json_object mode enforces valid JSON only, not schema conformance.
# The prompt must therefore carry the schema explicitly; otherwise the model
# invents field names (observed: course_name, instructors-as-objects, office_hours,
# grading-as-dict). Schema adds ~1500 tokens / request (~$0.0004 at V3 pricing).
_SCHEMA_JSON = json.dumps(to_strict_schema(SyllabusFacts), ensure_ascii=True, indent=2)

SYSTEM_PROMPT = (
    "You extract structured facts from university syllabus documents. "
    "Return JSON matching this schema exactly. Do not add fields that are not "
    "in the schema; do not rename fields; use null for any field the document "
    "does not state. Set due_date ONLY when the syllabus explicitly states a "
    "date — never infer or guess.\n\n"
    "JSON Schema:\n"
    f"{_SCHEMA_JSON}"
)


def build_user_prompt(
    text: str,
    pre_pass_entities: list[dict[str, Any]],
    sections: list[str],
    doc_id: str,
    source_url: str | None,
) -> str:
    pre_pass = {
        "regex_entities": pre_pass_entities[:80],
        "section_headers_detected": sections,
        "document_id": doc_id,
        "source_url": source_url,
    }
    return (
        "Extract SyllabusFacts from this syllabus.\n\n"
        f"Pre-pass features (regex/heuristic, may be noisy):\n"
        f"{json.dumps(pre_pass, ensure_ascii=True)}\n\n"
        f"Document text:\n{text}\n\n"
        "Return JSON only."
    )


def call_deepseek(client, model: str, system_prompt: str, user_prompt: str, max_retries: int = 3) -> tuple[str, int, int]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            return content, int(usage.prompt_tokens), int(usage.completion_tokens)
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"deepseek call failed after {max_retries} attempts: {last_err}")


def label_one(client, model: str, record: dict, max_text_chars: int, doc_id: str) -> dict:
    _, text = detect_text_field(record)
    text = text[:max_text_chars]
    regex_entities = [e.__dict__ for e in extract_regex_entities(text)]
    ner_entities = [e.__dict__ for e in extract_heuristic_ner(text)]
    sections = sorted({e["text"] for e in ner_entities if e["label"] == "SECTION"})

    user_prompt = build_user_prompt(text, regex_entities + ner_entities, sections, doc_id, record.get("source_url"))
    raw, in_tok, out_tok = call_deepseek(client, model, SYSTEM_PROMPT, user_prompt)

    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return {"ok": False, "error": "non_object_json", "raw": raw, "tokens": {"in": in_tok, "out": out_tok}}
        payload.setdefault("document_id", doc_id)
        if record.get("source_url"):
            payload.setdefault("source_url", record.get("source_url"))
        facts = SyllabusFacts.model_validate(payload)
        return {"ok": True, "facts": facts.model_dump(mode="json"), "tokens": {"in": in_tok, "out": out_tok}}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "raw": raw[:2000], "tokens": {"in": in_tok, "out": out_tok}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-label syllabi with DeepSeek-V3.")
    p.add_argument("--input-jsonl", default="data/ingested/us_freshman_core_syllabi_with_text.jsonl")
    p.add_argument("--output-jsonl", default="data/labeled/deepseek_facts.jsonl")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-base", default="https://api.deepseek.com")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--max-text-chars", type=int, default=20_000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip docs already in --output-jsonl.")
    return p.parse_args()


def _seen_ids(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        doc_id = (row.get("facts") or {}).get("document_id") or row.get("document_id")
        if doc_id:
            seen.add(str(doc_id))
    return seen


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set ${args.api_key_env} before running.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.api_base)
    rows = load_jsonl(args.input_jsonl, strict=False)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = _seen_ids(output) if args.resume else set()

    n_ok = n_fail = n_skip = 0
    total_in = total_out = 0
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as fh:
        for i, record in enumerate(rows, 1):
            if args.max_rows and (n_ok + n_fail) >= args.max_rows:
                break
            if not isinstance(record, dict) or not has_usable_document_text(record):
                n_skip += 1
                continue
            doc_id = normalize_record_id(record, i)
            if doc_id in seen:
                n_skip += 1
                continue

            try:
                result = label_one(client, args.model, record, args.max_text_chars, doc_id)
            except Exception as exc:
                result = {"ok": False, "error": f"call_failure:{exc}", "tokens": {"in": 0, "out": 0}}

            total_in += result["tokens"]["in"]
            total_out += result["tokens"]["out"]
            if result["ok"]:
                fh.write(json.dumps({"facts": result["facts"], "tokens": result["tokens"]}, ensure_ascii=True) + "\n")
                n_ok += 1
            else:
                fh.write(
                    json.dumps(
                        {"document_id": doc_id, "error": result["error"], "raw": result.get("raw", "")[:2000]},
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                n_fail += 1
            fh.flush()
            if i % 10 == 0:
                print(f"[deepseek] {i} processed: ok={n_ok} fail={n_fail} skip={n_skip}")

    # DeepSeek-V3 pricing reference (Nov 2025): $0.27 / M input, $1.10 / M output (cache miss).
    cost = (total_in / 1e6 * 0.27) + (total_out / 1e6 * 1.10)
    print(
        f"[deepseek] DONE ok={n_ok} fail={n_fail} skip={n_skip} | "
        f"tokens in={total_in} out={total_out} | est cost ~${cost:.2f}"
    )


if __name__ == "__main__":
    main()
