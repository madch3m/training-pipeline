"""Structured append-only error logging for fault-tolerant pipeline stages."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def log_pipeline_error(
    log_path: str | Path,
    *,
    stage: str,
    message: str,
    **extra: Any,
) -> None:
    """Append one JSON object per line (newline-delimited)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "message": message,
        **extra,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
