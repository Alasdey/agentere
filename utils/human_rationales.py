"""Human-written few-shot rationales, used in place of LLM-synthesized CoT.

One plain-text file per training document, in the directory named by
few_shot.human_rationales.dir. The document id is the file's stem (so
`1_15ecbplus.xml.txt` covers doc `1_15ecbplus.xml`), or the value of a leading
`Text:` / `Doc:` / `Document ID:` header line when there is one.

Everything after a line reading `Answer:` is the assistant turn the model will
imitate — reasoning followed by the final JSON array, exactly as the prompt asks
for it. Anything before that line (a copy of the document, notes) is ignored: the
document text the model sees always comes from the dataset. A file with no
`Answer:` line is taken as the answer in its entirety.

For a multi-step prompt (a `steps:` list in the prompt YAML), separate the turns
with `--- Step 2 ---` lines; one file with no separators means one turn.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from utils.runtime_config import register_reset

_ANSWER = re.compile(r"^[ \t]*(?:Answer|Response|Rationale)[ \t]*:[ \t]*$", re.M)
_HEADER = re.compile(r"^[ \t]*(?:Document ID|Doc|Text)[ \t]*:[ \t]*(\S+)[ \t]*$", re.M)
_STEP = re.compile(r"^[ \t]*-{2,}[ \t]*(?:CoT[ \t]+)?[Ss]tep[ \t]*\d+[ \t]*-{2,}[ \t]*$", re.M)

_CACHE: Optional[dict] = None
_CACHE_DIR: Optional[str] = None


def _reset() -> None:
    global _CACHE, _CACHE_DIR
    _CACHE = None
    _CACHE_DIR = None


register_reset(_reset)


def parse_file(path: Path) -> tuple[str, list]:
    """(doc_id, [one assistant turn per inference step])."""
    raw = path.read_text(encoding="utf-8")
    m = _ANSWER.search(raw)
    head, answer = (raw[:m.start()], raw[m.end():]) if m else ("", raw)

    doc_id = path.stem
    h = _HEADER.search(head if m else raw)
    if h:
        if not m:
            # A header with no `Answer:` line would feed the document text to the model as
            # part of its own answer. Refuse rather than corrupt the example silently.
            raise ValueError(
                f"[few_shot] {path} starts with a '{h.group(0).strip()}' header but has no "
                f"line reading 'Answer:' — add one before the reasoning, or drop the header "
                f"and let the file be the answer in its entirety"
            )
        doc_id = h.group(1)

    parts = [p.strip() for p in _STEP.split(answer)]
    steps = [p for p in parts if p] or [answer.strip()]
    return doc_id, steps


def load(dir_path: Optional[str]) -> dict:
    """{doc_id: [assistant turns]} for every file in dir_path. Cached per directory."""
    global _CACHE, _CACHE_DIR
    if not dir_path:
        return {}
    if _CACHE is not None and _CACHE_DIR == dir_path:
        return _CACHE

    d = Path(dir_path)
    out: dict = {}
    if d.is_dir():
        for path in sorted(d.iterdir()):
            if path.is_file() and path.suffix.lower() in (".txt", ".md") and not path.name.startswith("."):
                doc_id, steps = parse_file(path)
                out[doc_id] = steps
    _CACHE, _CACHE_DIR = out, dir_path
    if out:
        print(f"[few_shot] Loaded {len(out)} human-written rationale(s) from {dir_path}: "
              + ", ".join(sorted(out)))
    else:
        print(f"[few_shot] No human-written rationales found in {dir_path}")
    return out
