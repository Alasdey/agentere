"""
scripts/dev/causal_ranking_eval.py

Standalone eval of an LLM's ability to RANK mentions by causal relatedness.

For every event mention M ("target") in a document, the model is shown the
document (mentions tagged inline as <ID text>) and asked to order every other
mention in the document in descending order of probability of a causal
relation with M (either direction). The predicted order is scored against
gold causal pairs from MAVEN-ERE and EventStoryLine using Average Precision
and NDCG (binary relevance: gold-causal pair vs. not).

Targets with zero gold-causal partners are skipped — there is nothing to
rank correctly against, so AP/NDCG would be undefined for them.

Usage:
    python scripts/dev/causal_ranking_eval.py
    python scripts/dev/causal_ranking_eval.py --n-docs 15 --model-id deepseek/deepseek-v3.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sklearn.metrics import ndcg_score

from dataprep.dataprep import load_hf_dataset_parsed
from utils.labels import NOREL_VARIANTS
from utils.runtime_config import get_cfg

DATASET_KEYS = ["maven_ere", "event_story_line"]
DEFAULT_DOCS_PER_DATASET = 10  # 10 + 10 = 20 docs, the minimum the task asked for
DEFAULT_CONCURRENCY = 10
DEFAULT_MAX_MENTIONS = 600  # skip docs with more candidates than this (cost control)
RANDOM_SEED = 0

RANK_SYSTEM_PROMPT = (
    "You are an Event Causality Identification specialist.\n"
    "You are given a document with event mentions tagged inline as <ID text>,\n"
    "one TARGET mention, and a list of candidate mentions (every other mention\n"
    "in the document). Use the exact IDs given — never invent, renumber, or\n"
    "paraphrase one.\n\n"
    "Order ALL candidate IDs in descending order of the probability that they\n"
    "have a causal relation with the TARGET, in either direction (the target\n"
    "causing the candidate, or the candidate causing the target, both count).\n"
    "The candidate most likely to be a causal partner of the target goes first.\n\n"
    "Output ONLY a JSON array of candidate IDs, most likely first, with every\n"
    "candidate ID appearing exactly once. No markdown fences, no commentary."
)


def _user_prompt(doc_text: str, target_id: str, target_text: str, candidates: List[Tuple[str, str]]) -> str:
    candidate_lines = "\n".join(f'  - {cid} ("{ctext}")' for cid, ctext in candidates)
    return (
        f"Document:\n{doc_text}\n\n"
        f'TARGET mention: {target_id} ("{target_text}")\n\n'
        f"Candidate mentions:\n{candidate_lines}\n\n"
        "Output the ranked JSON array of candidate IDs now."
    )


def _gold_causal_pairs(gold_triples: List[Tuple[str, str, str]]) -> set:
    pairs = set()
    for src, label, tgt in gold_triples:
        if label.lower() in NOREL_VARIANTS:
            continue
        pairs.add(frozenset((src, tgt)))
    return pairs


async def _rank_target(
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    doc_text: str,
    target_id: str,
    target_text: str,
    candidates: List[Tuple[str, str]],
) -> Optional[List[str]]:
    async with semaphore:
        try:
            response = await llm.ainvoke([
                SystemMessage(content=RANK_SYSTEM_PROMPT),
                HumanMessage(content=_user_prompt(doc_text, target_id, target_text, candidates)),
            ])
            raw = response.content
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            ranked_ids = json.loads(match.group(0) if match else raw)
        except Exception:
            return None

    valid_ids = {cid for cid, _ in candidates}
    ranked_ids = [rid for rid in ranked_ids if isinstance(rid, str) and rid in valid_ids]
    # Candidates the model dropped from its output are pushed to the back,
    # in their original order, rather than discarded.
    seen = set(ranked_ids)
    ranked_ids += [cid for cid, _ in candidates if cid not in seen]
    return ranked_ids


def _score_ranking(ranked_ids: List[str], target_id: str, gold_pairs: set) -> Optional[Dict[str, float]]:
    relevance = [1 if frozenset((target_id, cid)) in gold_pairs else 0 for cid in ranked_ids]
    n_pos = sum(relevance)
    if n_pos == 0:
        return None

    hits = 0
    precisions = []
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    ap = sum(precisions) / n_pos

    descending_gains = [[len(relevance) - i for i in range(len(relevance))]]
    ndcg = ndcg_score([relevance], descending_gains)

    random_relevance = relevance[:]
    random.shuffle(random_relevance)
    random_hits = 0
    random_precisions = []
    for i, rel in enumerate(random_relevance, start=1):
        if rel:
            random_hits += 1
            random_precisions.append(random_hits / i)
    random_ap = sum(random_precisions) / n_pos
    random_ndcg = ndcg_score([random_relevance], descending_gains)

    return {
        "ap": ap,
        "ndcg": ndcg,
        "random_ap": random_ap,
        "random_ndcg": random_ndcg,
        "n_candidates": len(relevance),
        "n_positives": n_pos,
    }


async def evaluate_dataset(
    dataset_key: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    n_docs: int,
    max_mentions: int,
) -> List[Dict]:
    ds_cfg = get_cfg()["datasets"][dataset_key]
    loader = load_hf_dataset_parsed(
        repo_id=ds_cfg["repo_id"],
        split=ds_cfg["split"],
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        streaming=True,
        max_examples=n_docs,
    )

    tasks = []
    task_meta = []
    for doc in loader:
        mentions_map = doc["mentions_map"]
        if len(mentions_map) < 2 or len(mentions_map) > max_mentions:
            continue
        gold_pairs = _gold_causal_pairs(doc["gold_triples"])
        mention_items = list(mentions_map.items())

        for target_id, target_text in mention_items:
            candidates = [(cid, ctext) for cid, ctext in mention_items if cid != target_id]
            has_gold_partner = any(frozenset((target_id, cid)) in gold_pairs for cid, _ in candidates)
            if not has_gold_partner:
                continue
            tasks.append(_rank_target(llm, semaphore, doc["doc_text"], target_id, target_text, candidates))
            task_meta.append({"dataset": dataset_key, "doc_id": doc["id"], "target_id": target_id, "gold_pairs": gold_pairs})

    rankings = await asyncio.gather(*tasks)

    results = []
    for meta, ranked_ids in zip(task_meta, rankings):
        if ranked_ids is None:
            continue
        score = _score_ranking(ranked_ids, meta["target_id"], meta["gold_pairs"])
        if score is None:
            continue
        results.append({
            "dataset": meta["dataset"],
            "doc_id": meta["doc_id"],
            "target_id": meta["target_id"],
            "ranked_ids": ranked_ids,
            **score,
        })
    return results


def _summarize(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {"n_targets": 0, "n_docs": 0, "map": 0.0, "ndcg": 0.0, "random_map": 0.0, "random_ndcg": 0.0}
    n = len(results)
    return {
        "n_targets": n,
        "n_docs": len({(r["dataset"], r["doc_id"]) for r in results}),
        "map": sum(r["ap"] for r in results) / n,
        "ndcg": sum(r["ndcg"] for r in results) / n,
        "random_map": sum(r["random_ap"] for r in results) / n,
        "random_ndcg": sum(r["random_ndcg"] for r in results) / n,
    }


def _print_summary(label: str, summary: Dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    print(f"  docs evaluated:     {summary['n_docs']}")
    print(f"  targets evaluated:  {summary['n_targets']}")
    print(f"  MAP:                {summary['map']:.4f}  (random baseline: {summary['random_map']:.4f})")
    print(f"  mean NDCG:          {summary['ndcg']:.4f}  (random baseline: {summary['random_ndcg']:.4f})")


async def main_async(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)
    model_cfg = get_cfg().get("model", {})
    llm = ChatOpenAI(
        model=args.model_id or model_cfg.get("default_model_id"),
        temperature=0.0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=model_cfg.get("base_url", "https://openrouter.ai/api/v1"),
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    all_results: List[Dict] = []
    for dataset_key in DATASET_KEYS:
        print(f"Evaluating {dataset_key} ({args.n_docs} docs)...")
        results = await evaluate_dataset(dataset_key, llm, semaphore, args.n_docs, args.max_mentions)
        all_results.extend(results)
        _print_summary(dataset_key, _summarize(results))

    _print_summary("OVERALL", _summarize(all_results))

    out_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "causal_ranking_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{time.strftime('%Y-%m-%dT%H-%M-%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id or model_cfg.get("default_model_id"),
            "n_docs_per_dataset": args.n_docs,
            "results": all_results,
            "summary_by_dataset": {k: _summarize([r for r in all_results if r["dataset"] == k]) for k in DATASET_KEYS},
            "summary_overall": _summarize(all_results),
        }, f, indent=2)
    print(f"\nFull results written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-docs", type=int, default=DEFAULT_DOCS_PER_DATASET,
                         help="Docs to pull per dataset (default %(default)s; total across both datasets must be >= 20)")
    parser.add_argument("--model-id", type=str, default=None, help="Override config.yaml's model.default_model_id")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-mentions", type=int, default=DEFAULT_MAX_MENTIONS,
                         help="Skip documents with more mentions than this (cost control)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
