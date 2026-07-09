# tools/eci_extractor_intra.py
"""
Intra-sentence-only variant of `eci_extractor`.

Identical to `eci_extractor` in every respect — one orchestrator call, the
same per-target sub-calls on the same smaller model, the same aggregation and
final deduplicated/bidirectional output — EXCEPT it only ever evaluates pairs
of mentions that fall within the SAME sentence. Cross-sentence candidates are
dropped before the sub-call is issued, so cross-sentence relations can never be
predicted, never enter a sub-call's context, and never appear in the output.

Sentence membership is taken from the gold tokenization: the pipeline computes a
mention_id -> sentence-index map in dataprep and exposes it per-document via
`CURRENT_DOC_MENTION_SENTENCE` (see utils/context.py). Used by the MAVEN-ERE
intra-sentence-only setting (`maven_ere_standard_eci_intra_tool`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Tuple

from langchain_core.tools import tool

from utils.context import CURRENT_DOC_MENTION_SENTENCE

# Reuse eci_extractor's internals so both tools share one model config, one
# sub-call prompt/parser, and one trace-logging path.
from .eci_extractor import _extractor_cfg, _evaluate_target, _make_llm


@tool
async def eci_extractor_intra(document_text: str, mentions: List[Dict[str, str]]) -> str:
    """
    Intra-sentence causal relation extractor. Same contract as `eci_extractor`
    — call it EXACTLY ONCE per document, passing every event mention — but it
    only predicts relations between mentions that occur in the SAME sentence.
    Cross-sentence pairs are dropped internally, never sent to the model, and
    never appear in the output, so you can safely pass the full mention list.

    Its returned JSON array IS the final answer: already deduplicated, already
    bidirectional (causes + causedby). Forward it as-is rather than re-deriving
    or rewriting it.

    Args:
        document_text: The full document text with mentions tagged inline
                        as <ID text> (e.g. <e3 collapsed>).
        mentions: List of ALL event mentions in the document, each a dict
                  with keys "id" and "text".

    Returns:
        A JSON string: a flat array of
        {"pair": "A,B", "label": "causes"|"causedby"} objects.
    """
    if len(mentions) < 2:
        return json.dumps([])

    # mention_id -> sentence index, from the gold tokenization. If it is absent
    # (tool used outside the pipeline), _same_sentence is False for every pair
    # and the result is an empty array — no cross-sentence relations leak.
    sentence_of = CURRENT_DOC_MENTION_SENTENCE.get()

    def _same_sentence(a_id: str, b_id: str) -> bool:
        return (
            a_id in sentence_of
            and b_id in sentence_of
            and sentence_of[a_id] == sentence_of[b_id]
        )

    llm = _make_llm()
    concurrency = _extractor_cfg()["concurrency"]
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_evaluate(target: Dict[str, str], candidates: List[Dict[str, str]]):
        async with semaphore:
            return await _evaluate_target(llm, document_text, target, candidates)

    # Only same-sentence candidates per target. Targets with none get an empty
    # list -> _evaluate_target returns [] early without an LLM call.
    tasks = [
        _bounded_evaluate(
            target,
            [
                m
                for m in mentions
                if m["id"] != target["id"] and _same_sentence(target["id"], m["id"])
            ],
        )
        for target in mentions
    ]
    per_target_facts = await asyncio.gather(*tasks)

    # Aggregate into one fact per unordered pair (first sub-call to report a
    # pair wins, so two sub-calls disagreeing on direction can't both load
    # the canonical fact table with contradictory causers).
    canonical: Dict[frozenset, Tuple[str, str]] = {}
    for facts in per_target_facts:
        for causer, effect in facts:
            key = frozenset((causer, effect))
            canonical.setdefault(key, (causer, effect))

    relations = []
    for causer, effect in canonical.values():
        relations.append({"pair": f"{causer},{effect}", "label": "causes"})
        relations.append({"pair": f"{effect},{causer}", "label": "causedby"})

    return json.dumps(relations)
