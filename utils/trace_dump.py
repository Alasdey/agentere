from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

TRACE_PATH: Optional[Path] = None


def trace_dump(state) -> None:
    line = json.dumps(
        [msg.to_json() for msg in state["messages"]],
        default=str,
        ensure_ascii=False,
    ) + "\n"

    path = TRACE_PATH or Path("logs") / "raw_traces.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
