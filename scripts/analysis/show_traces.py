#!/usr/bin/env python3
"""
show_traces.py — pretty-print sampled LLM traces.

Each line in a .traces.jsonl file is a full message sequence for one
inference call (system + human + optional tool round-trips + final AI reply).

Usage:
    uv run scripts/analysis/show_traces.py                        # latest sample file
    uv run scripts/analysis/show_traces.py path/to/run.traces.jsonl
    uv run scripts/analysis/show_traces.py --all                  # every sample file
    uv run scripts/analysis/show_traces.py --trace 2              # only trace #2 (1-indexed)
    uv run scripts/analysis/show_traces.py --width 120            # wrap width
    uv run scripts/analysis/show_traces.py --no-color
    uv run scripts/analysis/show_traces.py --full                 # don't truncate content
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:    return _c("1", t)
def dim(t: str) -> str:     return _c("2", t)
def cyan(t: str) -> str:    return _c("96", t)
def green(t: str) -> str:   return _c("92", t)
def yellow(t: str) -> str:  return _c("93", t)
def magenta(t: str) -> str: return _c("95", t)
def blue(t: str) -> str:    return _c("94", t)
def red(t: str) -> str:     return _c("91", t)
def grey(t: str) -> str:    return _c("90", t)


# ── Helpers ───────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_LOGS_ROOT = _PROJECT_ROOT / "logs/allatonce/"

TRUNCATE_DEFAULT = 600   # chars before adding "[… truncated]"


def _msg_type(msg: Dict[str, Any]) -> str:
    """Return the bare class name, e.g. 'AIMessage'."""
    return msg.get("id", ["?"])[-1]


def _kwargs(msg: Dict[str, Any]) -> Dict[str, Any]:
    return msg.get("kwargs", {})


def _wrap(text: str, width: int, indent: str = "  ") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if len(line) <= width - len(indent):
            wrapped.append(indent + line)
        else:
            wrapped.extend(
                textwrap.wrap(line, width=width - len(indent),
                              initial_indent=indent, subsequent_indent=indent)
            )
    return "\n".join(wrapped)


def _truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit] + grey(f"  … [{len(text) - limit} chars truncated]")
    return text


# ── Per-message renderers ─────────────────────────────────────────────────────

def _render_system(kw: Dict, width: int, limit: int) -> str:
    content = _truncate(kw.get("content", ""), limit)
    header = bold(cyan("◆ SYSTEM"))
    return f"{header}\n{_wrap(content, width)}"


def _render_human(kw: Dict, width: int, limit: int) -> str:
    content = _truncate(kw.get("content", ""), limit)
    header = bold(green("▶ HUMAN"))
    return f"{header}\n{_wrap(content, width)}"


def _render_ai(kw: Dict, width: int, limit: int, idx: int) -> str:
    content = kw.get("content", "")
    tool_calls = kw.get("tool_calls", [])
    usage = kw.get("usage_metadata")

    parts = [bold(yellow(f"◀ AI  (step {idx})"))]

    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("args", {})
            args_str = json.dumps(args, ensure_ascii=False) if args else "(no args)"
            parts.append(
                f"  {bold(magenta('⚙ TOOL CALL'))}  {bold(name)}  {grey(args_str)}"
            )

    if content:
        parts.append(_wrap(_truncate(content, limit), width))

    if usage:
        inp = usage.get("input_tokens", "?")
        out = usage.get("output_tokens", "?")
        cache = usage.get("input_token_details", {}).get("cache_read", 0)
        cache_str = f"  cache_read={cache}" if cache else ""
        parts.append(grey(f"  tokens: {inp} in / {out} out{cache_str}"))

    return "\n".join(parts)


def _render_tool(kw: Dict, width: int, limit: int) -> str:
    content = _truncate(kw.get("content", ""), limit)
    tool_call_id = kw.get("tool_call_id", "")
    status = kw.get("status", "")
    status_str = f"  {grey(status)}" if status and status != "success" else ""
    header = f"{bold(blue('⬡ TOOL RESULT'))}  {grey(tool_call_id)}{status_str}"
    return f"{header}\n{_wrap(content, width)}"


def _render_message(msg: Dict, width: int, limit: int, ai_step: List[int]) -> str:
    mtype = _msg_type(msg)
    kw = _kwargs(msg)
    if mtype == "SystemMessage":
        return _render_system(kw, width, limit)
    if mtype == "HumanMessage":
        return _render_human(kw, width, limit)
    if mtype == "AIMessage":
        ai_step[0] += 1
        return _render_ai(kw, width, limit, ai_step[0])
    if mtype == "ToolMessage":
        return _render_tool(kw, width, limit)
    # fallback
    raw = json.dumps(kw, ensure_ascii=False)
    return f"{bold(red(f'? {mtype}'))}\n{_wrap(_truncate(raw, limit), width)}"


# ── Gold annotation lookup ────────────────────────────────────────────────────

def _load_gold_index(run_json_path: Path) -> Dict[str, Any]:
    """Load HF dataset referenced by the companion run JSON; return text→gold map."""
    import sys as _sys
    _project_root = run_json_path.parent.parent.parent
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    with open(run_json_path, encoding="utf-8") as f:
        run_data = json.load(f)

    cfg = run_data["config"]
    active_ds = cfg["active_dataset"]
    ds_cfg = cfg["datasets"][active_ds]
    repo_id    = ds_cfg["repo_id"]
    split      = ds_cfg.get("split", "test")
    text_field = ds_cfg.get("text_field", "text")
    ann_field  = ds_cfg.get("ann_field", "annots")

    from dataprep.dataprep import load_hf_dataset_parsed
    docs = list(load_hf_dataset_parsed(
        repo_id=repo_id, split=split,
        text_field=text_field, ann_field=ann_field,
        streaming=True,
    ))
    return {d["doc_text"]: d["gold_triples"] for d in docs}


def _extract_doc_text(trace: List[Dict]) -> str:
    """Pull the document text out of the last HumanMessage (after 'Text:\\n').

    Uses 'Pairs to classify' as the end-of-text boundary when present (handles
    multi-paragraph documents that contain blank lines inside the text).
    Falls back to the first blank line or end-of-string for prompts without
    an explicit pair list.
    """
    for msg in reversed(trace):
        if _msg_type(msg) == "HumanMessage":
            content = _kwargs(msg)["content"]
            m = re.search(r"Text:\n(.*?)(?=\n\nPairs to classify|\Z)", content, re.DOTALL)
            if not m:
                m = re.search(r"Text:\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    raise ValueError("Could not extract doc text from any HumanMessage in trace")


def _render_gold(gold_triples: List, width: int) -> str:
    header = bold(red("★ GOLD"))
    lines = [f"  {src} --[{lbl}]--> {tgt}" for src, lbl, tgt in gold_triples] or ["  (none)"]
    return f"{header}\n" + "\n".join(lines)


# ── Trace rendering ───────────────────────────────────────────────────────────

def render_trace(trace: List[Dict], trace_num: int, total: int,
                 width: int, limit: int,
                 gold_index: Dict[str, Any]) -> str:
    bar = "─" * width
    header = bold(f"{'─'*3} Trace {trace_num}/{total}  ({len(trace)} messages) {'─'*(width-30)}")
    ai_step: List[int] = [0]
    blocks = [render_message(m, width, limit, ai_step) for m in trace]

    doc_text = _extract_doc_text(trace)
    blocks.append(_render_gold(gold_index[doc_text], width))

    return f"\n{header}\n\n" + f"\n\n{dim(bar)}\n\n".join(blocks) + "\n"


def render_message(msg: Dict, width: int, limit: int, ai_step: List[int]) -> str:
    return _render_message(msg, width, limit, ai_step)


# ── File discovery ────────────────────────────────────────────────────────────

def find_sample_files(logs_root: Path) -> List[Path]:
    pattern = str(logs_root / "**" / "*.traces.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    return [Path(f) for f in files]


def latest_sample_file(logs_root: Path) -> Optional[Path]:
    files = find_sample_files(logs_root)
    return files[-1] if files else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pretty-print sampled LLM traces from .traces.jsonl files."
    )
    ap.add_argument(
        "file", nargs="?", default=None,
        help="Path to a .traces.jsonl file (default: latest in logs/)"
    )
    ap.add_argument("--all", action="store_true", help="Process every sample file found")
    ap.add_argument("--trace", type=int, default=None, metavar="N",
                    help="Show only trace N (1-indexed)")
    ap.add_argument("--width", type=int, default=100, help="Output width (default 100)")
    ap.add_argument("--full", action="store_true", help="Do not truncate message content")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")
    ap.add_argument("--logs", default=str(_LOGS_ROOT),
                    help="Root logs folder (for auto-discovery)")
    args = ap.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    limit = 0 if args.full else TRUNCATE_DEFAULT

    # Resolve files to process
    if args.all:
        files = find_sample_files(Path(args.logs))
        if not files:
            sys.exit(f"No .traces.jsonl files found under {args.logs}")
    elif args.file:
        files = [Path(args.file)]
    else:
        f = latest_sample_file(Path(args.logs))
        if f is None:
            sys.exit(f"No .traces.jsonl files found under {args.logs}")
        files = [f]

    for path in files:
        print(bold(f"\n{'═' * args.width}"))
        print(bold(f"  FILE: {path}"))
        print(bold(f"{'═' * args.width}"))

        stem = re.sub(r"\.traces\.jsonl$", "", path.name)
        companion_json = path.parent / f"{stem}.json"
        if not companion_json.exists():
            sys.exit(f"Companion run JSON not found: {companion_json}")
        gold_index = _load_gold_index(companion_json)
        print(grey(f"  Gold index loaded: {len(gold_index)} documents"))

        with open(path, encoding="utf-8") as fh:
            traces = [json.loads(line) for line in fh if line.strip()]

        if not traces:
            print(grey("  (empty file)"))
            continue

        indices = [args.trace] if args.trace else range(1, len(traces) + 1)

        for i in indices:
            if i < 1 or i > len(traces):
                print(red(f"  Trace {i} out of range (file has {len(traces)} traces)"))
                continue
            trace = traces[i - 1]
            print(render_trace(trace, i, len(traces), args.width, limit, gold_index=gold_index))


if __name__ == "__main__":
    main()
