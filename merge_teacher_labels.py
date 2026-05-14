"""Merge DeepSeek + GPT-4o + human-golden labels into one SyllabusFacts JSONL.

Resolution priority per document_id:
  golden  >  agreement (deepseek == gpt4o)  >  gpt4o (arbitrated)  >  gpt4o-only  >  deepseek-only

Each output row is::

  {"facts": <SyllabusFacts JSON>, "source": "<one of: golden|agreement|gpt4o_arbitrated|gpt4o_only|deepseek_only>"}

Stats are written alongside the merged JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Latest successful ``facts`` per document_id (later rows win)."""
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
        facts = row.get("facts")
        if not isinstance(facts, dict):
            continue
        doc_id = facts.get("document_id")
        if doc_id is None:
            continue
        out[str(doc_id)] = facts
    return out


def _normalize_for_compare(value: Any) -> Any:
    """Strip ordering noise so two equivalent labels compare equal."""
    if isinstance(value, list):
        normed = [_normalize_for_compare(v) for v in value]
        try:
            return sorted(normed, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=True))
        except TypeError:
            return normed
    if isinstance(value, dict):
        return {k: _normalize_for_compare(value[k]) for k in sorted(value)}
    return value


def labels_agree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _normalize_for_compare(a) == _normalize_for_compare(b)


def merge(
    deepseek: dict[str, dict[str, Any]],
    gpt4o: dict[str, dict[str, Any]],
    golden: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_ids = set(deepseek) | set(gpt4o) | set(golden)
    rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "golden": 0,
        "agreement": 0,
        "gpt4o_arbitrated": 0,
        "gpt4o_only": 0,
        "deepseek_only": 0,
    }
    for doc_id in sorted(all_ids):
        ds = deepseek.get(doc_id)
        gt = gpt4o.get(doc_id)
        gd = golden.get(doc_id)
        if gd is not None:
            rows.append({"facts": gd, "source": "golden"})
            stats["golden"] += 1
        elif gt is not None and ds is not None:
            if labels_agree(ds, gt):
                rows.append({"facts": gt, "source": "agreement"})
                stats["agreement"] += 1
            else:
                rows.append({"facts": gt, "source": "gpt4o_arbitrated"})
                stats["gpt4o_arbitrated"] += 1
        elif gt is not None:
            rows.append({"facts": gt, "source": "gpt4o_only"})
            stats["gpt4o_only"] += 1
        elif ds is not None:
            rows.append({"facts": ds, "source": "deepseek_only"})
            stats["deepseek_only"] += 1
    stats["total"] = len(rows)
    return rows, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge teacher labels into a single SyllabusFacts JSONL.")
    p.add_argument("--deepseek-jsonl", default="data/labeled/deepseek_facts.jsonl")
    p.add_argument(
        "--gpt4o-jsonl",
        default="data/labeled/gpt4o_arbitration.jsonl",
        help="May be either the arbitration output, the golden output, or a concatenation.",
    )
    p.add_argument("--golden-jsonl", default="data/labeled/golden_facts.jsonl")
    p.add_argument("--output-jsonl", default="data/labeled/teacher_labels.jsonl")
    p.add_argument("--stats-json", default="data/labeled/merge_stats.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_labels(Path(args.deepseek_jsonl))
    gt = load_labels(Path(args.gpt4o_jsonl))
    gd = load_labels(Path(args.golden_jsonl))

    rows, stats = merge(ds, gt, gd)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
