"""
Minimal, self-contained causal-relation-extraction eval.

One script: HF loading -> one LLM call per document -> JSON parsing -> binary
micro P/R/F1. No tools, no resampling, no few-shot, no langgraph — just enough
to sanity-check a model/prompt against the four datasets used elsewhere in this
repo (see dataprep/, main.py, utils/metrics.py for the "real" pipeline).

Event Story Line and Causal TimeBank have no held-out test split, so they are
evaluated with k-fold: each doc gets a fold id (doc_idx % n_folds) and we report
both pooled metrics and the mean/std across folds. MAVEN-ERE and MECI have a
real test split, so they run as a single fold.

Usage:
    OPENROUTER_API_KEY=... python scripts/experiments/simple_eval.py \
        --datasets meci causal_timebank --model qwen/qwen3-30b-a3b-instruct-2507 \
        --max-examples 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================================================
# DATASETS — one HF repo per task, mirroring config.yaml's `datasets:` block.
# =============================================================================
DATASETS: Dict[str, Dict[str, Any]] = {
    "maven_ere": {
        "repo_id": "Nofing/MAVEN-ERE-standard-eci",
        "split": "test",
        "constrain_to_pair_list": False,
        "n_folds": 1,
    },
    "meci": {
        "repo_id": "Nofing/MECI-v0.1-public-standard-eci",
        "split": "test",
        "constrain_to_pair_list": True,
        "n_folds": 1,
    },
    "causal_timebank": {
        "repo_id": "Nofing/CausalTimeBank-standard-eci",
        "split": "train",
        "constrain_to_pair_list": False,
        "n_folds": 10,
    },
    "event_story_line": {
        "repo_id": "Nofing/EventStoryLine-0.9-standard-eci",
        "split": "train",
        "constrain_to_pair_list": False,
        "n_folds": 5,
    },
}

NOREL_VARIANTS = frozenset({"norel", "none", "no_rel", "unknown"})
VALID_LABELS = frozenset({"causes", "causedby", "norel"})

RELATION_TRIPLE_RE = re.compile(
    r"<([^>\s]+)\s+[^>\n]+>\s+([A-Za-z0-9_\-]+)\s+<([^>\s]+)\s+[^>\n]+>"
)

SYSTEM_PROMPT = (
    "You extract causal relations between event mentions in a document.\n"
    "For each pair below, decide if the first event causes the second.\n"
    'Use exactly one label per pair: "causes", "causedby", or "norel".'
)

USER_TEMPLATE = """Text:
{doc_text}

Pairs to classify (use EXACT pair ids; output order does NOT matter):
{pair_lines}

Output a JSON array of objects with "pair" and "label" at the end of your answer.
Example: [{{"pair": "e1,e2", "label": "causes"}}]
"""

OPEN_ENDED_INSTRUCTION = (
    "Predict all causal event mention pairs in the text; omit a pair to mark it norel."
)


# =============================================================================
# LOADING
# =============================================================================
def load_documents(ds_cfg: Dict[str, Any], max_examples: int) -> List[Dict[str, Any]]:
    ds = load_dataset(ds_cfg["repo_id"], split=ds_cfg["split"])
    docs = []
    for i, row in enumerate(ds):
        if max_examples > 0 and i >= max_examples:
            break

        # Mirrors dataprep.load_hf_dataset_parsed's valid_labels filter so gold triples
        # match exactly what the main pipeline scores against (drops anything outside
        # the three canonical labels, e.g. stray annotation noise).
        gold_triples = [
            (m.group(1), m.group(2), m.group(3))
            for m in RELATION_TRIPLE_RE.finditer(row.get("annots", "") or "")
            if m.group(2) in VALID_LABELS
        ]

        tokens, spans, mentions = row.get("tokens", []), row.get("spans", []), row.get("mentions", [])
        mentions_map = {}
        if tokens and spans and mentions and len(spans) == len(mentions):
            for j, mention_id in enumerate(mentions):
                parts = [tokens[idx] for idx in spans[j] if 0 <= idx < len(tokens)]
                if parts:
                    mentions_map[mention_id] = " ".join(parts)

        pair_list_raw = row.get("pair_list")
        pair_list_ids: List[Tuple[str, str]] = []
        if mentions:
            if pair_list_raw is not None:
                for raw_a, raw_b in pair_list_raw:
                    a, b = int(raw_a), int(raw_b)
                    if a < len(mentions) and b < len(mentions):
                        pair_list_ids.append((mentions[a], mentions[b]))
            else:
                pair_list_ids = [
                    (mentions[a], mentions[b])
                    for a in range(len(mentions))
                    for b in range(len(mentions))
                    if a != b
                ]

        docs.append({
            "id": str(row.get("id", f"idx_{i}")),
            "doc_idx": i,
            "doc_text": row.get("text", ""),
            "gold_triples": gold_triples,
            "mentions_map": mentions_map,
            "pair_list_ids": pair_list_ids,
        })
    return docs


# =============================================================================
# PROMPTING + LLM CALL
# =============================================================================
def build_pair_lines(doc: Dict[str, Any], constrain_to_pair_list: bool, seed: int) -> str:
    # Only meci has constrain_to_pair_list=True, so this is the only branch that ever
    # puts an explicit pair list in the human prompt. It is always shuffled, with its own
    # Random(seed + doc_idx) instance so the shuffle is reproducible per-document
    # independent of asyncio scheduling/concurrency.
    if not (doc["pair_list_ids"] and constrain_to_pair_list):
        return OPEN_ENDED_INSTRUCTION
    mentions_map = doc["mentions_map"]
    seen, pairs = set(), []
    for src, tgt in doc["pair_list_ids"]:
        if (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        pairs.append((src, tgt))
    random.Random(seed + doc["doc_idx"]).shuffle(pairs)
    lines = [f'{src} ("{mentions_map.get(src, "")}"), {tgt} ("{mentions_map.get(tgt, "")}")' for src, tgt in pairs]
    return "\n".join(lines)


def parse_llm_json(content: str) -> List[Tuple[str, str, str]]:
    match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in output")
    data = json.loads(match.group(0))
    triples = []
    for item in data:
        pair = item.get("pair", "")
        if "," in pair:
            src, tgt = [p.strip() for p in pair.split(",", 1)]
            triples.append((src, item.get("label", "unknown"), tgt))
    return triples


async def call_llm_for_doc(llm: ChatOpenAI, doc: Dict[str, Any], constrain_to_pair_list: bool, seed: int, retries: int = 2) -> List[Tuple[str, str, str]]:
    doc_seed = seed + doc["doc_idx"]
    user_prompt = USER_TEMPLATE.format(
        doc_text=doc["doc_text"],
        pair_lines=build_pair_lines(doc, constrain_to_pair_list, seed),
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    # Per-doc API-level seed (OpenAI/OpenRouter "seed" param) for best-effort sampling
    # reproducibility — this only requests determinism from the provider, it doesn't
    # guarantee it (MoE kernels, provider routing on ":nitro" models, etc. can still vary).
    # Bumped by +1 on each retry so a retry isn't just replaying the exact same failed call.
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = await llm.bind(seed=doc_seed + attempt).ainvoke(messages)
            return parse_llm_json(resp.content)
        except Exception as e:
            last_error = e
    print(f"  [{doc['id']}] giving up after {retries + 1} attempts: {last_error}")
    return []


# =============================================================================
# METRICS — binary (any non-norel label = positive). Pair identity is the
# DIRECTED (src, tgt) tuple, matching utils/reporting.reconstruct_pairwise_predictions
# + utils/metrics.compute_binary_metrics under the repo's default data.binary_undirected:
# false. "causes(A,B)" and "causedby(B,A)" are scored as distinct pairs, same as main.py.
# =============================================================================
def positive_pairs(triples: List[Tuple[str, str, str]]) -> set:
    return {(src, tgt) for src, label, tgt in triples if label.lower() not in NOREL_VARIANTS}


def pair_counts(gold_triples: List[Tuple[str, str, str]], pred_triples: List[Tuple[str, str, str]]) -> Dict[str, int]:
    """Per-doc TP/FP/FN counts only — no precision/recall/F1 here. Those are only
    meaningful once TP/FP/FN are pooled across the whole dataset (see micro_prf1),
    exactly like utils.metrics.compute_binary_metrics operating on the full
    gold/pred label arrays in main.py. Computing F1 per document and then
    averaging is a different (and not framework-equivalent) metric."""
    gold_pos, pred_pos = positive_pairs(gold_triples), positive_pairs(pred_triples)
    tp = len(gold_pos & pred_pos)
    fp = len(pred_pos - gold_pos)
    fn = len(gold_pos - pred_pos)
    return {"tp": tp, "fp": fp, "fn": fn}


def micro_prf1(counts: List[Dict[str, int]]) -> Dict[str, float]:
    """Pools TP/FP/FN across all docs first, then derives precision/recall/F1 once —
    matching main.py, where kfold only gates few-shot exclusion and metrics are always
    computed globally over the full set of docs, never split and averaged per fold."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# RUN — one independent session per dataset.
# =============================================================================
async def run_dataset(name: str, args: argparse.Namespace) -> None:
    ds_cfg = DATASETS[name]
    print(f"\n=== [{name}] {ds_cfg['repo_id']} (split={ds_cfg['split']}) ===")

    docs = load_documents(ds_cfg, args.max_examples)
    print(f"Loaded {len(docs)} documents.")

    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        api_key=os.environ.get("OPENROUTER_API_KEY", "sk-fake-key-for-testing"),
        base_url="https://openrouter.ai/api/v1",
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def process(doc):
        async with semaphore:
            pred_triples = await call_llm_for_doc(llm, doc, ds_cfg["constrain_to_pair_list"], args.seed)
        counts = pair_counts(doc["gold_triples"], pred_triples)
        fold = doc["doc_idx"] % ds_cfg["n_folds"]
        print(f"  [{name}] [fold {fold}] doc {doc['id']}: tp={counts['tp']} fp={counts['fp']} fn={counts['fn']}")
        return counts

    counts = await asyncio.gather(*(process(doc) for doc in docs))

    # Single global pooled metric over every doc, regardless of fold — matching main.py,
    # where doc_idx % n_folds only gates which docs are eligible as few-shot examples
    # and never splits scoring. There is no kfold-CV-style fold averaging here.
    global_metrics = micro_prf1(counts)
    print(f"\n[{name}] precision={global_metrics['precision']:.4f}  recall={global_metrics['recall']:.4f}  micro_f1={global_metrics['f1']:.4f}  (n_docs={len(docs)}, n_folds={ds_cfg['n_folds']})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified single-call causal-extraction eval.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--model", default="qwen/qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in args.datasets:
        await run_dataset(name, args)  # each dataset gets its own LLM client/session


if __name__ == "__main__":
    asyncio.run(main())
