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
import json
import random
from pathlib import Path
from typing import FrozenSet, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from dataprep.dataprep import load_hf_dataset_parsed
from utils.formatting import format_pair_lines, format_gold_output
from utils.context import CURRENT_DOC_TEXT, CURRENT_DOC_MENTIONS, CURRENT_DOC_FOLD
import utils.trace_dump as trace_dump

from utils.runtime_config import get_cfg, register_reset

# ── Lazy caches ───────────────────────────────────────────────────────────────

_TRAIN_CACHE: Optional[list] = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_MENTION_SETS: Optional[list] = None  # frozenset[str] per doc, parallel to _TRAIN_CACHE
_BERT_MODEL = None
_BERT_EMBEDDINGS = None  # np.ndarray, shape (n_docs, hidden), parallel to _TRAIN_CACHE
_COT_CACHE: dict = {}  # doc_id -> generated CoT string


def _reset_caches() -> None:
    global _TRAIN_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS
    global _BERT_MODEL, _BERT_EMBEDDINGS, _COT_CACHE
    _TRAIN_CACHE = None
    _TFIDF_VECTORIZER = None
    _TFIDF_MATRIX = None
    _MENTION_SETS = None
    _BERT_MODEL = None
    _BERT_EMBEDDINGS = None
    _COT_CACHE = {}


register_reset(_reset_caches)

_COT_STEP1_PROMPT = (
    "The correct label assignments for this document are:\n\n{gold_json}\n\n"
    "Write a complete response exactly as the system prompt requires — all steps followed by "
    "the final JSON — as if you reasoned your way to these labels from the text alone. "
    "Your reasoning must lead naturally to each label. "
    "Do not mention or imply that you already know the answer."
)

_COT_STEP2_PROMPT = (
    "Rewrite your response from scratch to remove all privileged knowledge. "
    "Replace any phrasing that reveals foreknowledge "
    "(e.g. \"we see in the JSON that X is NoRel\", \"the answer shows\", "
    "\"as indicated by the labels\", \"we know that\", \"the expected output\") "
    "with genuine reasoning grounded in the text and rules. "
    "Every sentence in every step should read as if you discovered the label through analysis, "
    "not confirmed it from a pre-known answer. Output only the rewritten response, nothing else."
)


def preload() -> None:
    """Eagerly load and fit the training cache. Call once before concurrent inference."""
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = _load_train_split()


def _load_train_split() -> list:
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS, _BERT_MODEL, _BERT_EMBEDDINGS

    cfg = get_cfg()
    fs_cfg = cfg.get("few_shot", {})
    active_ds = cfg["active_dataset"]
    ds_cfg = cfg["datasets"][active_ds]

    split = fs_cfg.get("split", "train")
    repo_id = fs_cfg.get("repo_id") or ds_cfg["repo_id"]

    data_cfg = cfg["data"]
    docs = list(load_hf_dataset_parsed(
        repo_id=repo_id,
        split=split,
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        valid_labels=set(data_cfg["labels"]),
        streaming=True,
        binary_undirected=data_cfg["binary_undirected"],
        shuffle_pair_list=data_cfg["shuffle_pair_list"],
    ))

    pool_size = fs_cfg.get("pool_size")
    if pool_size:
        docs = docs[:pool_size]

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
    cfg = get_cfg()
    active_ds = cfg["active_dataset"]
    binary_undirected = cfg["data"]["binary_undirected"]
    constrain_to_pair_list = cfg["datasets"][active_ds]["constrain_to_pair_list"]
    parts = []
    for i, doc in enumerate(docs, 1):
        pair_lines = format_pair_lines(doc, binary_undirected=binary_undirected, constrain_to_pair_list=constrain_to_pair_list)
        gold_out = format_gold_output(doc["gold_triples"], pair_list_ids=doc.get("pair_list_ids"))
        parts.append(
            f"--- Example {i} ---\n"
            f"Text:\n{doc['doc_text']}\n\n"
            f"Pairs:\n{pair_lines}\n\n"
            f"Output:\n{gold_out}"
        )
    return "\n\n".join(parts)


async def _generate_cot_for_doc(
    doc: dict,
    system_prompt: str,
    user_content: str,
    gold_output: str,
    graph_ainvoke,
) -> str:
    """Two-step LLM generation: produce a realistic CoT+answer using the gold labels, then
    rewrite to strip any privileged-knowledge phrasing so the result looks like genuine reasoning."""
    doc_id = doc.get("id", "")
    if doc_id in _COT_CACHE:
        return _COT_CACHE[doc_id]

    step1_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
        HumanMessage(content=_COT_STEP1_PROMPT.format(gold_json=gold_output)),
    ]
    state1 = await graph_ainvoke(step1_messages)
    trace_dump.trace_dump(state1)
    ai_step1 = state1["messages"][-1].content

    step2_messages = step1_messages + [
        AIMessage(content=ai_step1),
        HumanMessage(content=_COT_STEP2_PROMPT),
    ]
    state2 = await graph_ainvoke(step2_messages)
    trace_dump.trace_dump(state2)
    ai_step2 = state2["messages"][-1].content

    _COT_CACHE[doc_id] = ai_step2
    return ai_step2


async def get_few_shot_message_pairs(
    user_template: str,
    active_ds: str,
    system_prompt: str = "",
    graph_ainvoke=None,
) -> list:
    """Returns [(human_content, ai_content), ...] for conversation-based few-shot injection.

    When few_shot.cot_generation.enabled is true and graph_ainvoke is provided, the ai_content
    is generated via two LLM calls that produce a realistic CoT+answer from the gold labels,
    then rewrite it to remove any privileged-knowledge phrasing.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    cfg = get_cfg()
    n = cfg["few_shot"]["n_examples"]
    docs = _select_examples(_TRAIN_CACHE, n)

    cot_enabled = (
        cfg["few_shot"]["cot_generation"]["enabled"]
        and graph_ainvoke is not None
        and system_prompt
    )

    binary_undirected = cfg["data"]["binary_undirected"]
    constrain_to_pair_list = cfg["datasets"][active_ds]["constrain_to_pair_list"]
    pairs = []
    for doc in docs:
        pair_lines = format_pair_lines(doc, binary_undirected=binary_undirected, constrain_to_pair_list=constrain_to_pair_list)
        human_content = user_template.format(
            doc_text=doc["doc_text"],
            pair_lines=pair_lines,
            doc_id=doc.get("id", ""),
        )
        gold_output = format_gold_output(doc["gold_triples"], pair_list_ids=doc.get("pair_list_ids"))
        if cot_enabled:
            ai_content = await _generate_cot_for_doc(
                doc, system_prompt, human_content, gold_output, graph_ainvoke
            )
        else:
            ai_content = gold_output
        pairs.append((human_content, ai_content))
    return pairs


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _select_examples(docs: list, n: int) -> list:
    cfg = get_cfg()
    fs_cfg = cfg.get("few_shot", {})
    selection = fs_cfg.get("selection")

    # Exclude same-fold docs when running k-fold CV.
    # Keep track of original indices so similarity matrices can be sliced correctly.
    current_fold = CURRENT_DOC_FOLD.get()
    active_ds = cfg["active_dataset"]
    n_folds = cfg["datasets"][active_ds]["kfold"]["n_folds"]
    if current_fold >= 0 and n_folds > 1:
        filtered = [(i, d) for i, d in enumerate(docs) if d["doc_idx"] % n_folds != current_fold]
        orig_indices, docs = zip(*filtered) if filtered else ([], [])
        orig_indices = list(orig_indices)
        docs = list(docs)
    else:
        orig_indices = list(range(len(docs)))

    if selection == "smallest":
        non_empty = [d for d in docs if len(d.get("pair_list_ids", [])) != 0]
        return sorted(non_empty, key=lambda d: len(d["pair_list_ids"]))[:n]
    
    if selection == "similarity" and _TFIDF_VECTORIZER is not None:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query = CURRENT_DOC_TEXT.get()
        if query:
            scores = cosine_similarity(
                _TFIDF_VECTORIZER.transform([query]),
                _TFIDF_MATRIX[orig_indices]
            )[0]
            top_local = np.argsort(scores)[::-1][:n]
            return [docs[i] for i in top_local]

    elif selection == "mentions" and _MENTION_SETS is not None:
        query_mentions = CURRENT_DOC_MENTIONS.get()
        if query_mentions:
            scores = [_jaccard(query_mentions, _MENTION_SETS[i]) for i in orig_indices]
            top_local = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
            return [docs[i] for i in top_local]

    elif selection == "bert" and _BERT_MODEL is not None and _BERT_EMBEDDINGS is not None:
        import numpy as np

        query = CURRENT_DOC_TEXT.get()
        if query:
            q_emb = _BERT_MODEL.encode([query], convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb, keepdims=True).clip(min=1e-12)
            scores = (_BERT_EMBEDDINGS[orig_indices] @ q_emb.T).ravel()
            top_local = np.argsort(scores)[::-1][:n]
            return [docs[i] for i in top_local]

    return random.sample(docs, min(n, len(docs)))


# ── CoT disk cache ────────────────────────────────────────────────────────────

def _load_cot_disk_cache(cache_path: str) -> None:
    path = Path(cache_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        _COT_CACHE.update(json.load(f))
    print(f"[few_shot] Loaded {len(_COT_CACHE)} CoT entries from {cache_path}")


def _save_cot_disk_cache(cache_path: str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_COT_CACHE, f, ensure_ascii=False, indent=2)
    print(f"[few_shot] Saved {len(_COT_CACHE)} CoT entries to {cache_path}")


# ── Pre-generation ────────────────────────────────────────────────────────────

async def pregenerate_cot(
    test_docs: list,
    user_template: str,
    active_ds: str,
    system_prompt: str,
    graph_ainvoke,
    concurrency: int = 10,
    cache_path: Optional[str] = None,
) -> None:
    """Generate CoT for every unique few-shot training doc needed across all test_docs.

    Runs before the main inference loop so each training doc is processed exactly once.
    Results land in _COT_CACHE; inference calls to _generate_cot_for_doc then just hit
    the cache. Optionally persists the cache to disk to avoid re-generating across runs.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    if cache_path:
        _load_cot_disk_cache(cache_path)

    cfg = get_cfg()
    n = cfg["few_shot"]["n_examples"]
    active_ds = cfg["active_dataset"]
    kfold_cfg = cfg["datasets"][active_ds]["kfold"]
    n_folds = kfold_cfg["n_folds"] if kfold_cfg["enabled"] else 1

    # Scan all test docs to collect the unique training docs that will be used as few-shots.
    needed: dict = {}  # training doc_id -> doc
    for test_doc in test_docs:
        CURRENT_DOC_TEXT.set(test_doc["doc_text"])
        CURRENT_DOC_MENTIONS.set(frozenset(test_doc.get("mentions_map", {}).values()))
        CURRENT_DOC_FOLD.set(test_doc["doc_idx"] % n_folds if n_folds > 1 else -1)
        for fs_doc in _select_examples(_TRAIN_CACHE, n):
            doc_id = fs_doc.get("id", "")
            if doc_id not in _COT_CACHE and doc_id not in needed:
                needed[doc_id] = fs_doc

    if not needed:
        print("[few_shot] All CoT entries already cached — skipping pre-generation.")
        return

    print(f"[few_shot] Pre-generating CoT for {len(needed)} unique few-shot docs...")
    sem = asyncio.Semaphore(concurrency)

    binary_undirected = cfg["data"]["binary_undirected"]
    constrain_to_pair_list = cfg["datasets"][active_ds]["constrain_to_pair_list"]

    async def _gen(doc):
        async with sem:
            pair_lines = format_pair_lines(doc, binary_undirected=binary_undirected, constrain_to_pair_list=constrain_to_pair_list)
            human_content = user_template.format(
                doc_text=doc["doc_text"],
                pair_lines=pair_lines,
                doc_id=doc.get("id", ""),
            )
            gold_output = format_gold_output(doc["gold_triples"], pair_list_ids=doc.get("pair_list_ids"))
            await _generate_cot_for_doc(doc, system_prompt, human_content, gold_output, graph_ainvoke)

    await asyncio.gather(*[_gen(doc) for doc in needed.values()])
    print(f"[few_shot] CoT pre-generation done. Cache size: {len(_COT_CACHE)}")

    if cache_path:
        _save_cot_disk_cache(cache_path)


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

    n = get_cfg().get("few_shot", {}).get("n_examples", 3)
    sample = _select_examples(_TRAIN_CACHE, n)
    return _format_examples(sample)
