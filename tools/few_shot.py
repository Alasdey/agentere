"""
Tool that retrieves labelled examples from the training split.
Selection modes, controlled by few_shot.selection in config.yaml:
  "random"     — resample every call (default)
  "similarity" — TF-IDF cosine on document text (sklearn required)
  "mentions"   — Jaccard similarity on the sets of event mention texts
The full training split is loaded lazily and cached in memory.
"""
from __future__ import annotations

import asyncio
import os
import random
import yaml
from contextvars import ContextVar
from typing import FrozenSet, Optional

from langchain_core.tools import tool

from dataprep.dataprep import load_hf_dataset_parsed
from utils.formatting import format_pair_lines, format_gold_output

# ── Context variables — set by main.py, read by the tool ─────────────────────

CURRENT_DOC_TEXT: ContextVar[str] = ContextVar("current_doc_text", default="")
CURRENT_DOC_MENTIONS: ContextVar[FrozenSet[str]] = ContextVar("current_doc_mentions", default=frozenset())

# ── Module-level config ───────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

# ── Lazy caches ───────────────────────────────────────────────────────────────

_TRAIN_CACHE: Optional[list] = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_MENTION_SETS: Optional[list] = None  # frozenset[str] per doc, parallel to _TRAIN_CACHE


def _load_train_split() -> list:
    global _TRAIN_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS

    fs_cfg = _CFG.get("few_shot", {})
    active_ds = _CFG["active_dataset"]
    ds_cfg = _CFG["datasets"][active_ds]

    split = fs_cfg.get("split", "train")
    repo_id = fs_cfg.get("repo_id") or ds_cfg["repo_id"]

    docs = list(load_hf_dataset_parsed(
        repo_id=repo_id,
        split=split,
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        streaming=True,
    ))
    _TRAIN_CACHE = docs

    selection = fs_cfg.get("selection")

    if selection == "similarity":
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [d["doc_text"] for d in docs]
        _TFIDF_VECTORIZER = TfidfVectorizer()
        _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(texts)

    elif selection == "mentions":
        _MENTION_SETS = [
            frozenset(d.get("mentions_map", {}).values())
            for d in docs
        ]

    return docs


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


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _select_examples(docs: list, n: int) -> list:
    fs_cfg = _CFG.get("few_shot", {})
    selection = fs_cfg.get("selection")

    if selection == "similarity" and _TFIDF_VECTORIZER is not None:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query = CURRENT_DOC_TEXT.get()
        if query:
            scores = cosine_similarity(
                _TFIDF_VECTORIZER.transform([query]),
                _TFIDF_MATRIX
            )[0]
            top_indices = np.argsort(scores)[::-1][:n]
            return [docs[i] for i in top_indices]

    elif selection == "mentions" and _MENTION_SETS is not None:
        query_mentions = CURRENT_DOC_MENTIONS.get()
        if query_mentions:
            scores = [_jaccard(query_mentions, ms) for ms in _MENTION_SETS]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
            return [docs[i] for i in top_indices]

    return random.sample(docs, min(n, len(docs)))


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool
async def few_shot_examples(comment: str = "") -> str:
    """
    Retrieves labelled examples from the training set to ground your predictions.
    Call this before analysing the document.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    n = _CFG.get("few_shot", {}).get("n_examples", 3)
    sample = _select_examples(_TRAIN_CACHE, n)
    return _format_examples(sample)
