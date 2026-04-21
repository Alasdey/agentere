"""
Tool that retrieves random labelled examples from the training split.
Examples are re-sampled on every invocation for variance across documents.
The full training split is loaded lazily and cached in memory.
"""
from __future__ import annotations

import asyncio
import os
import random
import yaml
from typing import Optional

from langchain_core.tools import tool

from dataprep.dataprep import load_hf_dataset_parsed
from utils.formatting import format_pair_lines, format_gold_output

# ── Module-level config (loaded at import, like other tools) ─────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

# ── Lazy cache for the training split ────────────────────────────────────────

_TRAIN_CACHE: Optional[list] = None


def _load_train_split() -> list:
    fs_cfg = _CFG.get("few_shot", {})
    active_ds = _CFG["active_dataset"]
    ds_cfg = _CFG["datasets"][active_ds]

    split = fs_cfg.get("split", "train")
    repo_id = fs_cfg.get("repo_id") or ds_cfg["repo_id"]

    return list(load_hf_dataset_parsed(
        repo_id=repo_id,
        split=split,
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        streaming=True,
    ))


def _format_examples(docs: list) -> str:
    active_ds = _CFG["active_dataset"]
    parts = []
    for i, doc in enumerate(docs, 1):
        pair_lines = format_pair_lines(doc, active_ds)
        gold_out = format_gold_output(doc["gold_triples"], active_ds)
        parts.append(
            f"--- Example {i} ---\n"
            f"Text:\n{doc['doc_text']}\n\n"
            f"Pairs:\n{pair_lines}\n\n"
            f"Output:\n{gold_out}"
        )
    return "\n\n".join(parts)


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool
async def few_shot_examples(comment: str = "") -> str:
    """
    Retrieves labelled examples from the training set to ground your predictions.
    Examples are randomly selected — call this before analysing the document.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    n = _CFG.get("few_shot", {}).get("n_examples", 3)
    sample = random.sample(_TRAIN_CACHE, min(n, len(_TRAIN_CACHE)))
    return _format_examples(sample)
