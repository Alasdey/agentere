#!/usr/bin/env python3
"""
Offline CoT cache generator.

For each document in the training split, runs a two-step LLM pipeline:
  1. Generate step-by-step reasoning for every gold-labelled pair.
  2. Scrub any premature label mentions from the reasoning body so the
     answer only appears as the final "→ <label>" conclusion line.

Results are saved to:
  cot_cache/{active_dataset}_{split}.json   (keyed by doc_id)

Already-cached docs are skipped on re-run (incremental).

Usage:
  uv run python scripts/generate_cot.py
  uv run python scripts/generate_cot.py --concurrency 8 --max-docs 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dataprep.dataprep import load_hf_dataset_parsed
from utils.config import load_config
from utils.formatting import format_pair_lines

# =============================================================================
# Prompts
# =============================================================================

_STEP1_SYSTEM = """\
You are an expert annotation assistant for event relation extraction.
Given a text, a set of event pairs, and their gold labels, generate detailed \
step-by-step reasoning explaining why each pair has its assigned label.

For each pair follow these steps in order:
1. **Locate** both events in the text. Quote the exact span and note which sentence it is in.
2. **Explicit signals**: identify any causal or explanatory connective directly linking \
the two events ("because", "due to", "leading to", "as a result", "sparked", "enabled", \
"prevented", etc.). Quote it if present.
3. **Implicit evidence**: if no explicit signal, look for — \
(a) counterfactual dependency (would one event have occurred without the other?), \
(b) a concrete mechanism described in the text, \
(c) enablement or prevention patterns.
4. **Counter-evidence**: note any factor arguing against a relation \
(temporal-only link, reporting verb, confounding event, pure world knowledge).
5. **Conclusion**: state "→ <label>" as the very last line of the pair's block. \
Do not state the label anywhere else.

Write one block per pair, separated by a blank line.\
"""

_STEP2_SYSTEM = """\
You are editing a chain-of-thought reasoning trace for event relation extraction.
Your task: in each pair's reasoning block, ensure the final label \
(e.g. "→ Rel", "→ NoRel", "→ CauseEffect", "→ PRECONDITION") appears ONLY \
as the very last line of that block.

Remove or rephrase any sentence in the analysis body that prematurely states \
or implies the conclusion — e.g.:
  "I can see that A causes B"
  "this is clearly a CauseEffect relation"
  "since the relation is Rel"
  "confirming that it is a PRECONDITION"

The body should describe evidence and reasoning WITHOUT naming the answer \
until the final "→ <label>" line.

Return all reasoning blocks with the same structure; only remove premature \
conclusion statements. Do not add new content.\
"""


# =============================================================================
# Core
# =============================================================================

async def _generate_cot(doc: dict, llm: ChatOpenAI, active_ds: str) -> str:
    """Two-step CoT generation for one document. Returns the cleaned reasoning string."""
    pairs_with_labels = "\n".join(
        f"({src}, {tgt}): {lbl}" for src, lbl, tgt in doc["gold_triples"]
    )

    # Only include explicit pair list for closed-set datasets
    pair_lines = format_pair_lines(doc, active_ds)
    has_pair_list = active_ds == "meci"
    pairs_section = f"Pairs to classify:\n{pair_lines}\n\n" if has_pair_list else ""

    user1 = (
        f"Text:\n{doc['doc_text']}\n\n"
        f"{pairs_section}"
        f"Gold labels:\n{pairs_with_labels}\n\n"
        "Generate step-by-step reasoning for each pair."
    )

    resp1 = await llm.ainvoke([SystemMessage(content=_STEP1_SYSTEM), HumanMessage(content=user1)])

    resp2 = await llm.ainvoke([
        SystemMessage(content=_STEP2_SYSTEM),
        HumanMessage(content=f"Reasoning to clean:\n\n{resp1.content}"),
    ])
    return resp2.content


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    active_ds = config["active_dataset"]
    ds_cfg = config["datasets"][active_ds]
    fs_cfg = config.get("few_shot", {})

    split = fs_cfg.get("split", "train")
    repo_id = fs_cfg.get("repo_id") or ds_cfg["repo_id"]

    # Output path
    cache_dir = Path(__file__).resolve().parents[1] / "cot_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{active_ds}_{split}.json"

    # Load existing cache
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached CoTs from {cache_path}")

    # Load training split
    print(f"Loading {repo_id} ({split})...")
    docs = list(load_hf_dataset_parsed(
        repo_id=repo_id,
        split=split,
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        streaming=True,
    ))
    if args.max_docs:
        docs = docs[:args.max_docs]

    pending = [d for d in docs if d["id"] not in cache]
    print(f"{len(docs)} docs total — {len(docs) - len(pending)} cached, {len(pending)} to generate")

    if not pending:
        print("Nothing to do.")
        return

    # Build LLM
    model_cfg = config["model"]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    llm = ChatOpenAI(
        model=model_cfg["default_model_id"],
        temperature=0.0,
        api_key=api_key,
        base_url=model_cfg.get("base_url", "https://openrouter.ai/api/v1"),
    )

    # Generate with concurrency limit
    sem = asyncio.Semaphore(args.concurrency)
    errors: list[str] = []

    async def process(doc: dict) -> None:
        async with sem:
            try:
                cot = await _generate_cot(doc, llm, active_ds)
                cache[doc["id"]] = cot
                print(f"  ✓ {doc['id']}")
            except Exception as e:
                errors.append(doc["id"])
                print(f"  ✗ {doc['id']}: {e}")

    await asyncio.gather(*[process(doc) for doc in pending])

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(cache)} CoTs → {cache_path}")
    if errors:
        print(f"Failed ({len(errors)}): {errors}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CoT cache for few-shot examples.")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Number of concurrent LLM requests (default: 5)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Limit number of documents to process (for testing)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
