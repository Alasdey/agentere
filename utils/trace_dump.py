from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional

TRACE_PATH: Optional[Path] = None
_sample_written: int = 0
SAMPLE_LIMIT: int = 5


def trace_dump(state) -> None:
    global _sample_written

    line = json.dumps(
        [msg.to_json() for msg in state["messages"]],
        default=str,
        ensure_ascii=False,
    ) + "\n"

    path = TRACE_PATH or Path("logs") / "raw_traces.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "ab") as f:
        f.write(line.encode("utf-8"))

    if _sample_written < SAMPLE_LIMIT:
        sample_path = path.with_name(path.name.replace(".jsonl.gz", ".sample.jsonl"))
        with open(sample_path, "a", encoding="utf-8") as f:
            f.write(line)
        _sample_written += 1
