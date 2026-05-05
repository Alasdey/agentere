"""
Tool that retrieves labelled examples from the training split.
Selection modes, controlled by few_shot.selection in config.yaml:
  "random"     — resample every call (default)
  "similarity" — TF-IDF cosine on document text (sklearn required)
  "mentions"   — Jaccard similarity on the sets of event mention texts
  "bert"       — cosine similarity on sentence-transformer embeddings (sentence-transformers required)
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
_BERT_MODEL = None
_BERT_EMBEDDINGS = None  # np.ndarray, shape (n_docs, hidden), parallel to _TRAIN_CACHE
_COT_CACHE: Optional[dict] = None  # doc_id → reasoning string, loaded from cot_cache/


def preload() -> None:
    """Eagerly load and fit the training cache. Call once before concurrent inference."""
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = _load_train_split()


async def select_examples_async(n: int) -> list:
    """Public async interface: ensure cache is loaded, then select n examples."""
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)
    return _select_examples(_TRAIN_CACHE, n)


def get_cot_cache() -> Optional[dict]:
    """Return the loaded CoT cache (None if use_cot is disabled or cache not loaded yet)."""
    return _COT_CACHE


def _load_cot_cache() -> None:
    global _COT_CACHE
    import json
    from pathlib import Path
    fs_cfg = _CFG.get("few_shot", {})
    if not fs_cfg.get("use_cot", False):
        return
    active_ds = _CFG["active_dataset"]
    split = fs_cfg.get("split", "train")
    cache_path = Path(_CONFIG_PATH).parent / "cot_cache" / f"{active_ds}_{split}.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CoT cache not found at {cache_path}. "
            f"Run: uv run python scripts/generate_cot.py"
        )
    _COT_CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(_COT_CACHE)} CoTs from {cache_path}")


def _load_train_split() -> list:
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS, _BERT_MODEL, _BERT_EMBEDDINGS
    _load_cot_cache()

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

    elif selection == "bert":
        import numpy as np
        from sentence_transformers import SentenceTransformer

        import torch
        model_name = fs_cfg.get("bert_model", "all-MiniLM-L6-v2")
        _BERT_MODEL = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        texts = [d["doc_text"] for d in docs]
        _BERT_EMBEDDINGS = _BERT_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        norms = np.linalg.norm(_BERT_EMBEDDINGS, axis=1, keepdims=True)
        _BERT_EMBEDDINGS = _BERT_EMBEDDINGS / np.where(norms == 0, 1, norms)

    return docs


def _format_examples(docs: list) -> str:
    active_ds = _CFG["active_dataset"]
    binary_mode = _CFG.get("binary_mode", False)
    parts = []
    for i, doc in enumerate(docs, 1):
        pair_lines = format_pair_lines(doc, active_ds)
        gold_out = format_gold_output(doc["gold_triples"], active_ds, binary_mode=binary_mode)
        cot = _COT_CACHE.get(doc["id"]) if _COT_CACHE else None
        reasoning_block = f"Reasoning:\n{cot}\n\n" if cot else ""
        parts.append(
            f"--- Example {i} ---\n"
            f"Text:\n{doc['doc_text']}\n\n"
            f"Pairs:\n{pair_lines}\n\n"
            f"{reasoning_block}"
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

    elif selection == "bert" and _BERT_MODEL is not None and _BERT_EMBEDDINGS is not None:
        import numpy as np

        query = CURRENT_DOC_TEXT.get()
        if query:
            q_emb = _BERT_MODEL.encode([query], convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb, keepdims=True).clip(min=1e-12)
            scores = (_BERT_EMBEDDINGS @ q_emb.T).ravel()
            top_indices = np.argsort(scores)[::-1][:n]
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
