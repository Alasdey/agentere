import json
from pathlib import Path

def trace_dump(state):
    """Dump raw traces to a separate file for inspection"""
    
    # Dump full trace
    traces_path = Path("logs") / "raw_traces.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_path, "a", encoding="utf-8") as f:
        f.write(json.dumps([msg.to_json() for msg in state["messages"]], default=str, ensure_ascii=False) + "\n")

    print(f"Raw traces written to: {traces_path}")

    return