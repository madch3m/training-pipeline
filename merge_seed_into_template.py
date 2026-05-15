"""Merge a GPT-4o-seeded SyllabusFacts JSONL with the template's _hint blocks.

Reads:
- ``--template-jsonl``  the original template (carries ``_hint`` per row).
- ``--seed-jsonl``      output of ``label_with_gpt4o.py --mode golden`` (carries
                        ``facts`` + ``tokens``; ``tokens`` is debugging metadata).

Writes:
- ``--output-jsonl``    one row per template document, with the seed's ``facts``
                        block merged in and the template's ``_hint`` preserved.
                        Documents the seed failed to label keep an empty
                        ``facts`` block (so the labeler can be re-run for misses).

Iteration loop: edit the output in place, save, run ``validate_golden_facts.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _index_by_doc_id(path: Path, key: str = "facts") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            continue
        block = row.get(key)
        if not isinstance(block, dict):
            continue
        doc_id = block.get("document_id")
        if doc_id is None:
            continue
        out[str(doc_id)] = block
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge GPT-4o seed into template _hint blocks.")
    p.add_argument("--template-jsonl", default="data/labeled/golden_facts_template.jsonl")
    p.add_argument("--seed-jsonl", default="data/labeled/golden_facts_seed.jsonl")
    p.add_argument("--output-jsonl", default="data/labeled/golden_facts.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    template_rows: list[dict[str, Any]] = []
    for line in Path(args.template_jsonl).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        template_rows.append(json.loads(s))

    seed_facts = _index_by_doc_id(Path(args.seed_jsonl), key="facts")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_seeded = n_unseeded = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in template_rows:
            facts = row.get("facts") or {}
            doc_id = facts.get("document_id")
            seeded = seed_facts.get(str(doc_id)) if doc_id else None
            if seeded:
                merged_facts = seeded
                n_seeded += 1
            else:
                merged_facts = facts
                n_unseeded += 1
            fh.write(json.dumps({"facts": merged_facts, "_hint": row.get("_hint", {})}, ensure_ascii=True) + "\n")

    print(f"[merge_seed] wrote {n_seeded + n_unseeded} rows → {out_path} (seeded={n_seeded}, unseeded={n_unseeded})")


if __name__ == "__main__":
    main()
