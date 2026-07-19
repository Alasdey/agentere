"""
Tool that retrieves labelled examples from the training split.
Selection modes, controlled by few_shot.selection in config.yaml:
  "random"     — resample every call (default)
  "similarity" — TF-IDF cosine on document text (sklearn required)
  "mentions"   — Jaccard similarity on the sets of event mention texts
  "bert"       — cosine similarity on sentence-transformer embeddings (sentence-transformers required)
few_shot.density_stratify (bool) layers on top of any selection mode: instead of taking the n
best/random docs outright, it buckets the eligible pool into n quantile bins by density (causal
relations / mentions) and takes one doc per bin, so the chosen set always spans low-to-high
density. Combined with resampling.distinct_fewshots, each resample run's own few-shot set still
spans the full density range — see get_few_shot_message_pairs' round-robin split, which this
feeds into unchanged.
few_shot.intra_only (bool) restricts gold relations to only same-sentence (intra-sentence) gold
triples, per doc["mention_sentence"], everywhere gold is shown to the model: the
_format_examples/few_shot_examples tool's "Output:", the CoT-synthesis "correct labels are"
hint, and the non-CoT "Output:" fallback used when cot_generation.enabled is false. It does not
affect doc selection.
The full training split is loaded lazily and cached in memory.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import FrozenSet, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from dataprep.dataprep import load_hf_dataset_parsed
from utils.formatting import format_pair_lines, format_gold_output
from utils.context import CURRENT_DOC_TEXT, CURRENT_DOC_MENTIONS, CURRENT_DOC_FOLD
from utils.labels import NOREL_VARIANTS
import utils.trace_dump as trace_dump

from utils.runtime_config import get_cfg, register_reset

# ── Lazy caches ───────────────────────────────────────────────────────────────

_TRAIN_CACHE: Optional[list] = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_MENTION_SETS: Optional[list] = None  # frozenset[str] per doc, parallel to _TRAIN_CACHE
_BERT_MODEL = None
_BERT_EMBEDDINGS = None  # np.ndarray, shape (n_docs, hidden), parallel to _TRAIN_CACHE
_DOC_DENSITIES: Optional[list] = None  # float per doc (causal relations / mentions), parallel to _TRAIN_CACHE
_COT_CACHE: dict = {}  # doc_id -> generated CoT string
_DUMP_RUN_DIR: Optional[Path] = None  # <dump_dir>/<run timestamp>, lazily created on first dump this run


def _reset_caches() -> None:
    global _TRAIN_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS
    global _BERT_MODEL, _BERT_EMBEDDINGS, _DOC_DENSITIES, _COT_CACHE, _DUMP_RUN_DIR
    _TRAIN_CACHE = None
    _TFIDF_VECTORIZER = None
    _TFIDF_MATRIX = None
    _MENTION_SETS = None
    _BERT_MODEL = None
    _BERT_EMBEDDINGS = None
    _DOC_DENSITIES = None
    _COT_CACHE = {}
    _DUMP_RUN_DIR = None


register_reset(_reset_caches)

_COT_GOLD_HINT = (
    "The correct label assignments for this document are:\n\n{gold_json}\n\n"
    "Write your response for this step exactly as required — reasoning that leads naturally "
    "to these labels from the text alone. Do not mention or imply that you already know the answer."
)

_COT_GOLD_HINT_CONTINUE = (
    "Continue. The correct labels are still:\n\n{gold_json}\n\n"
    "Write your response for this step, consistent with the gold labels and your prior reasoning. "
    "Do not reveal foreknowledge."
)

_COT_DECONTAM_PROMPT = (
    "Rewrite your response from scratch to remove all privileged knowledge. "
    "Replace any phrasing that reveals foreknowledge "
    "(e.g. \"we see in the JSON that X is NoRel\", \"the answer shows\", "
    "\"as indicated by the labels\", \"we know that\", \"the expected output\") "
    "with genuine reasoning grounded in the text and rules. "
    "Every sentence should read as if you discovered the label through analysis, "
    "not confirmed it from a pre-known answer. Output only the rewritten response, nothing else."
)

_COT_CORRECT_HINT = (
    "The correct label assignments for this document are:\n\n{gold_json}\n\n"
    "Your response above was written without access to these labels. Revise it so the "
    "reasoning leads naturally to the correct labels — fix any wrong conclusions, add "
    "missing evidence, and correct the final answer. Keep anything that already agrees "
    "with the correct labels. Do not mention or imply that you were given the answer; "
    "write as though this is a refined version of your own analysis. "
    "Output the full revised response, nothing else."
)


def preload() -> None:
    """Eagerly load and fit the training cache. Call once before concurrent inference."""
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = _load_train_split()


def _doc_density(doc: dict) -> float:
    """Causal relations (label not in NOREL_VARIANTS) per event mention. 0.0 if the doc has
    no mentions at all."""
    n_mentions = len(doc.get("mentions_map", {}))
    if n_mentions == 0:
        return 0.0
    n_causal = sum(1 for _, label, _ in doc["gold_triples"] if label.lower() not in NOREL_VARIANTS)
    return n_causal / n_mentions


def _load_train_split() -> list:
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _MENTION_SETS, _BERT_MODEL, _BERT_EMBEDDINGS, _DOC_DENSITIES

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

    if fs_cfg.get("density_stratify"):
        _DOC_DENSITIES = [_doc_density(d) for d in docs]

    return docs


def _format_examples(docs: list) -> str:
    cfg = get_cfg()
    active_ds = cfg["active_dataset"]
    binary_undirected = cfg["data"]["binary_undirected"]
    constrain_to_pair_list = cfg["datasets"][active_ds]["constrain_to_pair_list"]
    intra_only = cfg["few_shot"]["intra_only"]
    parts = []
    for i, doc in enumerate(docs, 1):
        pair_lines = format_pair_lines(doc, binary_undirected=binary_undirected, constrain_to_pair_list=constrain_to_pair_list)
        gold_triples = doc["gold_triples"]
        if intra_only:
            gold_triples = _filter_same_sentence(gold_triples, doc["mention_sentence"])
        gold_out = format_gold_output(gold_triples, pair_list_ids=doc.get("pair_list_ids"))
        parts.append(
            f"--- Example {i} ---\n"
            f"Text:\n{doc['doc_text']}\n\n"
            f"Pairs:\n{pair_lines}\n\n"
            f"Output:\n{gold_out}"
        )
    return "\n\n".join(parts)


def _filter_same_sentence(gold_triples: list, mention_sentence: dict) -> list:
    """Keep only gold triples whose two mentions fall in the same sentence, per
    doc["mention_sentence"]. Mirrors eci_extractor_intra.py's _same_sentence check,
    but takes the per-doc map directly since callers here already hold the doc dict.
    """
    return [
        (src, lbl, tgt)
        for src, lbl, tgt in gold_triples
        if src in mention_sentence
        and tgt in mention_sentence
        and mention_sentence[src] == mention_sentence[tgt]
    ]


async def _generate_cot_for_doc(
    doc: dict,
    system_prompt: str,
    user_content: str,
    gold_output: str,
    graph_ainvoke,
    steps: list = None,
    num_steps: int = 2,
    intra_only: bool = False,
    dump_dir: Optional[str] = None,
) -> list:
    """Generate a realistic CoT for each inference step, then decontaminate.

    Single-call mode (no `steps`) supports two protocols, selected by `num_steps`:
      2 (default): gold-guided generation -> decontaminate.
      3: blind generation (no golds) -> correct with golds -> decontaminate.
    Multi-step mode (task-level `steps` list) always uses the 2-step gold-guided protocol,
    regardless of `num_steps`.

    `gold_output` is expected to already be pre-filtered by the caller when few_shot.intra_only
    is set — this function does not itself filter it. `intra_only` here exists purely to
    namespace the _COT_CACHE key so that CoT generated from filtered vs. unfiltered gold can
    never collide in the cache.

    Returns a list of clean AI responses — one per inference step (length 1 for single-call mode,
    length len(steps)+1 for multi-step mode). Results are cached in _COT_CACHE.
    """
    doc_id = doc.get("id", "")
    cache_key = f"{doc_id}::n{num_steps}" + ("::intra" if intra_only else "")
    if cache_key in _COT_CACHE:
        return _COT_CACHE[cache_key]

    gold_hint = _COT_GOLD_HINT.format(gold_json=gold_output)
    gold_hint_cont = _COT_GOLD_HINT_CONTINUE.format(gold_json=gold_output)

    if not steps:
        # ── Single-call mode ──────────────────────────────────────────────────
        if num_steps == 3:
            # Step 1: blind generation, no gold visibility.
            gen_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            state = await graph_ainvoke(gen_messages)
            trace_dump.trace_dump(state)
            blind_raw = state["messages"][-1].content

            # Step 2: correct the blind reasoning using the golds.
            correct_hint = _COT_CORRECT_HINT.format(gold_json=gold_output)
            correct_messages = gen_messages + [
                AIMessage(content=blind_raw),
                HumanMessage(content=correct_hint),
            ]
            state2 = await graph_ainvoke(correct_messages)
            trace_dump.trace_dump(state2)
            raw = state2["messages"][-1].content
            pre_decontam_messages = correct_messages + [AIMessage(content=raw)]
        else:
            # Step 1 (2-step protocol): gold-guided generation.
            gen_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
                HumanMessage(content=gold_hint),
            ]
            state = await graph_ainvoke(gen_messages)
            trace_dump.trace_dump(state)
            raw = state["messages"][-1].content
            pre_decontam_messages = gen_messages + [AIMessage(content=raw)]

        # Final step: decontaminate.
        decontam_messages = pre_decontam_messages + [
            HumanMessage(content=_COT_DECONTAM_PROMPT),
        ]
        state_final = await graph_ainvoke(decontam_messages)
        trace_dump.trace_dump(state_final)
        clean = [state_final["messages"][-1].content]
    else:
        # ── Multi-step mode ───────────────────────────────────────────────────
        # Build guided generation conversation step by step, then decontaminate each.
        # Conversation grows: [Sys, Human(user), Human(gold_hint)] → AI(step0_raw)
        #   → Human(step1.prompt), Human(gold_hint_cont) → AI(step1_raw) → ...
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
            HumanMessage(content=gold_hint),
        ]
        raw_responses = []

        # First step response (user_content already incorporates step 1 instruction)
        state = await graph_ainvoke(messages)
        trace_dump.trace_dump(state)
        raw_responses.append(state["messages"][-1].content)
        messages.append(AIMessage(content=raw_responses[-1]))

        for step in steps:
            messages.append(HumanMessage(content=step["prompt"]))
            messages.append(HumanMessage(content=gold_hint_cont))
            state = await graph_ainvoke(messages)
            trace_dump.trace_dump(state)
            raw_responses.append(state["messages"][-1].content)
            messages.append(AIMessage(content=raw_responses[-1]))

        # Decontaminate each raw response independently
        clean = []
        for raw in raw_responses:
            decontam_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=raw),
                HumanMessage(content=_COT_DECONTAM_PROMPT),
            ]
            state = await graph_ainvoke(decontam_messages)
            trace_dump.trace_dump(state)
            clean.append(state["messages"][-1].content)

    if dump_dir:
        _dump_cot_to_disk(doc, clean, dump_dir)

    _COT_CACHE[cache_key] = clean
    return clean


def _n_fewshot_sets(cfg) -> int:
    """One distinct few-shot set per resample run when resampling.distinct_fewshots
    is enabled, else a single set shared by every run."""
    rs_cfg = cfg["experiment"]["resampling"]
    if rs_cfg["enabled"] and rs_cfg["n_runs"] > 1 and rs_cfg["distinct_fewshots"]:
        return rs_cfg["n_runs"]
    return 1


async def get_few_shot_message_pairs(
    user_template: str,
    active_ds: str,
    steps: list = None,
    system_prompt: str = "",
    graph_ainvoke=None,
) -> list:
    """Returns few-shot example sets for conversation-based injection.

    Return type: List[List[List[Tuple[str, str]]]]
      Outermost list: one example set per resample run when resampling.distinct_fewshots
        is enabled (length n_runs), else a single shared set (length 1).
      Middle list: one entry per few-shot example.
      Inner list: (human, ai) exchange pairs for that example.
        - Single-call mode (no steps): one (human, ai) pair per example.
        - Multi-step mode: (human_step1, ai_step1), (step2_prompt, ai_step2), ... per example.

    When few_shot.cot_generation.enabled is true and graph_ainvoke is provided, AI content is
    generated via guided generation from gold labels then decontaminated. When few_shot.intra_only
    is also true, the gold shown to the synthesis LLM — and the non-CoT "Output:" fallback used
    when cot_generation.enabled is false — is first restricted to same-sentence gold triples only.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    cfg = get_cfg()
    n = cfg["few_shot"]["n_examples"]
    n_sets = _n_fewshot_sets(cfg)
    docs = _select_examples(_TRAIN_CACHE, n * n_sets)
    if n_sets > 1 and len(docs) < n * n_sets:
        raise ValueError(
            f"[few_shot] resampling.distinct_fewshots needs {n * n_sets} examples "
            f"({n_sets} runs x {n}) but only {len(docs)} eligible docs are available"
        )

    cot_enabled = (
        cfg["few_shot"]["cot_generation"]["enabled"]
        and graph_ainvoke is not None
        and system_prompt
    )
    num_steps = cfg["few_shot"]["cot_generation"]["num_steps"]
    intra_only = cfg["few_shot"]["intra_only"]
    dump_dir = cfg["few_shot"]["cot_generation"].get("dump_dir")

    binary_undirected = cfg["data"]["binary_undirected"]
    constrain_to_pair_list = cfg["datasets"][active_ds]["constrain_to_pair_list"]
    all_examples = []
    for doc in docs:
        pair_lines = format_pair_lines(doc, binary_undirected=binary_undirected, constrain_to_pair_list=constrain_to_pair_list)
        human_content = user_template.format(
            doc_text=doc["doc_text"],
            pair_lines=pair_lines,
            doc_id=doc.get("id", ""),
        )
        gold_triples = doc["gold_triples"]
        if intra_only:
            gold_triples = _filter_same_sentence(gold_triples, doc["mention_sentence"])
        gold_output = format_gold_output(gold_triples, pair_list_ids=doc.get("pair_list_ids"))

        if cot_enabled:
            # Returns List[str]: one clean response per step (len=1 for single-call)
            cot_responses = await _generate_cot_for_doc(
                doc, system_prompt, human_content, gold_output, graph_ainvoke,
                steps=steps, num_steps=num_steps, intra_only=intra_only,
                dump_dir=dump_dir,
            )
        else:
            # No CoT: first step gets gold output, continuation steps get empty string
            n_steps = len(steps) + 1 if steps else 1
            cot_responses = [gold_output] + [""] * (n_steps - 1)

        if not steps:
            # Single-call: one (human, ai) pair
            example_exchanges = [(human_content, cot_responses[0])]
        else:
            # Multi-step: step 0 uses user_content, then continuation prompts for steps[1:]
            example_exchanges = [(human_content, cot_responses[0])]
            for step, ai_resp in zip(steps, cot_responses[1:]):
                example_exchanges.append((step["prompt"], ai_resp))

        all_examples.append(example_exchanges)
    # Split round-robin so each run's set spans the selection ranking evenly
    # (docs are ordered best-first for similarity/mentions/bert selections, or
    # ascending-density-bin order when few_shot.density_stratify is enabled).
    return [all_examples[i::n_sets] for i in range(n_sets)]


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _relevance_scores(docs: list, orig_indices: list):
    """Query-relevance score per doc (parallel to `docs`) for the active few_shot.selection
    strategy. Returns None when that strategy has no query-based ranking (e.g. "random",
    "smallest") or no query is set yet — callers should fall back to a random tie-break."""
    cfg = get_cfg()
    selection = cfg.get("few_shot", {}).get("selection")

    if selection == "similarity" and _TFIDF_VECTORIZER is not None:
        from sklearn.metrics.pairwise import cosine_similarity

        query = CURRENT_DOC_TEXT.get()
        if query:
            return cosine_similarity(
                _TFIDF_VECTORIZER.transform([query]),
                _TFIDF_MATRIX[orig_indices]
            )[0]

    elif selection == "mentions" and _MENTION_SETS is not None:
        query_mentions = CURRENT_DOC_MENTIONS.get()
        if query_mentions:
            return [_jaccard(query_mentions, _MENTION_SETS[i]) for i in orig_indices]

    elif selection == "bert" and _BERT_MODEL is not None and _BERT_EMBEDDINGS is not None:
        import numpy as np

        query = CURRENT_DOC_TEXT.get()
        if query:
            q_emb = _BERT_MODEL.encode([query], convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb, keepdims=True).clip(min=1e-12)
            return (_BERT_EMBEDDINGS[orig_indices] @ q_emb.T).ravel()

    return None


def _select_density_stratified(docs: list, orig_indices: list, n: int) -> list:
    """Pick up to n docs spanning the training pool's density range (causal relations per
    mention), one per quantile bin, ranked within each bin by `_relevance_scores` (random
    tie-break when that returns None). Returned in ascending-density bin order — this ordering
    is what makes get_few_shot_message_pairs' round-robin split spread density evenly across
    resample runs when resampling.distinct_fewshots is enabled."""
    order = sorted(range(len(docs)), key=lambda i: _DOC_DENSITIES[orig_indices[i]])
    n_bins = min(n, len(order))
    bin_bounds = [round(b * len(order) / n_bins) for b in range(n_bins + 1)]
    scores = _relevance_scores(docs, orig_indices)

    selected = []
    for b in range(n_bins):
        bin_idxs = order[bin_bounds[b]:bin_bounds[b + 1]] or order
        pick = max(bin_idxs, key=lambda i: scores[i]) if scores is not None else random.choice(bin_idxs)
        selected.append(pick)
    return [docs[i] for i in selected]


def _select_examples(docs: list, n: int) -> list:
    cfg = get_cfg()
    fs_cfg = cfg.get("few_shot", {})
    selection = fs_cfg.get("selection")

    # Exclude same-fold docs when running k-fold CV.
    # Keep track of original indices so similarity matrices (and density lookups) can be
    # sliced correctly.
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

    if fs_cfg.get("density_stratify"):
        return _select_density_stratified(docs, orig_indices, n)

    if selection == "smallest":
        non_empty = [d for d in docs if len(d.get("pair_list_ids", [])) != 0]
        return sorted(non_empty, key=lambda d: len(d["pair_list_ids"]))[:n]

    scores = _relevance_scores(docs, orig_indices)
    if scores is not None:
        top_local = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:n]
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


def _dump_run_dir(dump_dir: str) -> Path:
    """<dump_dir>/<run timestamp>/, created once per run (first dump call after the
    last _reset_caches) and reused for every doc dumped in that same run."""
    global _DUMP_RUN_DIR
    if _DUMP_RUN_DIR is None:
        _DUMP_RUN_DIR = Path(dump_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        _DUMP_RUN_DIR.mkdir(parents=True, exist_ok=True)
    return _DUMP_RUN_DIR


def _dump_cot_to_disk(doc: dict, clean: list, dump_dir: str) -> None:
    """Write one synthesized CoT to <dump_dir>/<run timestamp>/<doc_id>.txt for manual
    inspection, alongside the source document text. Overwrites on every fresh generation
    within the same run (i.e. a cache hit in _generate_cot_for_doc does not re-write the
    file); a new run gets its own timestamped folder, so reruns never clobber past dumps."""
    doc_id = str(doc.get("id", "unknown"))
    safe_id = re.sub(r"[^\w.-]", "_", doc_id) or "unknown"
    path = _dump_run_dir(dump_dir) / f"{safe_id}.txt"
    cot_text = "\n\n".join(
        f"--- CoT step {i} ---\n{step}" for i, step in enumerate(clean, 1)
    ) if len(clean) > 1 else clean[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"Document ID: {doc_id}\n\n"
            f"--- Document text ---\n{doc['doc_text']}\n\n"
            f"--- Synthesized CoT ---\n{cot_text}\n"
        )


# ── Pre-generation ────────────────────────────────────────────────────────────

async def pregenerate_cot(
    test_docs: list,
    user_template: str,
    active_ds: str,
    system_prompt: str,
    graph_ainvoke,
    concurrency: int = 10,
    cache_path: Optional[str] = None,
    steps: list = None,
    num_steps: int = 2,
    retries: int = 3,
    dump_dir: Optional[str] = None,
) -> None:
    """Generate CoT for every unique few-shot training doc needed across all test_docs.

    Runs before the main inference loop so each training doc is processed exactly once.
    Results land in _COT_CACHE; inference calls to _generate_cot_for_doc then just hit
    the cache. Optionally persists the cache to disk to avoid re-generating across runs.

    Each doc's generation is retried on failure (transient provider errors, e.g. malformed
    API responses) like the main inference loop, and the disk cache is saved incrementally
    after each success so a late failure doesn't discard already-generated CoTs.
    """
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        _TRAIN_CACHE = await asyncio.to_thread(_load_train_split)

    if cache_path:
        _load_cot_disk_cache(cache_path)

    cfg = get_cfg()
    n = cfg["few_shot"]["n_examples"] * _n_fewshot_sets(cfg)
    active_ds = cfg["active_dataset"]
    kfold_cfg = cfg["datasets"][active_ds]["kfold"]
    n_folds = kfold_cfg["n_folds"] if kfold_cfg["enabled"] else 1
    intra_only = cfg["few_shot"]["intra_only"]

    # Scan all test docs to collect the unique training docs that will be used as few-shots.
    needed: dict = {}  # training doc_id -> doc
    for test_doc in test_docs:
        CURRENT_DOC_TEXT.set(test_doc["doc_text"])
        CURRENT_DOC_MENTIONS.set(frozenset(test_doc.get("mentions_map", {}).values()))
        CURRENT_DOC_FOLD.set(test_doc["doc_idx"] % n_folds if n_folds > 1 else -1)
        for fs_doc in _select_examples(_TRAIN_CACHE, n):
            doc_id = fs_doc.get("id", "")
            cache_key = f"{doc_id}::n{num_steps}" + ("::intra" if intra_only else "")
            if cache_key not in _COT_CACHE and doc_id not in needed:
                needed[doc_id] = fs_doc

    if not needed:
        print("[few_shot] All CoT entries already cached — skipping pre-generation.")
        return

    print(f"[few_shot] Pre-generating CoT for {len(needed)} unique few-shot docs...")
    sem = asyncio.Semaphore(concurrency)
    save_lock = asyncio.Lock()

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
            gold_triples = doc["gold_triples"]
            if intra_only:
                gold_triples = _filter_same_sentence(gold_triples, doc["mention_sentence"])
            gold_output = format_gold_output(gold_triples, pair_list_ids=doc.get("pair_list_ids"))

            last_error = None
            for attempt in range(retries + 1):
                try:
                    await _generate_cot_for_doc(
                        doc, system_prompt, human_content, gold_output, graph_ainvoke,
                        steps=steps, num_steps=num_steps, intra_only=intra_only,
                        dump_dir=dump_dir,
                    )
                    break
                except Exception as e:
                    last_error = e
                    print(f"[few_shot] CoT generation failed for doc {doc.get('id', '')} (attempt {attempt}): {e}")
                    await asyncio.sleep(1)
            else:
                raise ValueError(f"[few_shot] CoT generation for doc {doc.get('id', '')} failed after {retries} retries: {last_error}")

            if cache_path:
                async with save_lock:
                    _save_cot_disk_cache(cache_path)

    await asyncio.gather(*[_gen(doc) for doc in needed.values()])
    print(f"[few_shot] CoT pre-generation done. Cache size: {len(_COT_CACHE)}")


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
