# tools/eci_extractor.py
"""
Causal relation extractor — a leaner alternative to `eci`.

`eci` is called once PER MENTION *by the orchestrator*, re-sending the full
document text and the full candidate list on every call, and returns verbose
per-candidate reasoning that all stays in the orchestrator's context for the
rest of the conversation. The orchestrator then has to re-derive the final
deduplicated, bidirectional relation list from memory across however many
tool calls that took. On dense documents (MAVEN-ERE/CausalTimeBank have docs
with 90-275 mentions) that means the orchestrator's context fills with
mostly-irrelevant reasoning and it has to recall hundreds of relations from
it at the end.

This tool is called exactly ONCE by the orchestrator. Internally, it still
evaluates one target mention against all other mentions per sub-call (the
per-mention granularity is what makes exhaustive coverage of dense documents
tractable for a single LLM call), but those sub-calls and their verbose
intermediate output happen entirely inside this tool, run concurrently, and
are aggregated here. Only the final deduplicated, bidirectional relation
list — never the per-mention scratch work — ever enters the orchestrator's
context.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.runtime_config import get_cfg, register_reset
from utils import trace_dump
from utils import trace_dump

# Deliberately a different (and smaller) model than the orchestrator's
# default_model_id — each sub-call is a focused binary-plus-direction
# judgement, not multi-step agentic reasoning, so a 30B instruct model is
# plenty and much cheaper to run per mention than the orchestrator's model.
# DEFAULT_EXTRACTOR_MODEL_ID = "qwen/qwen3-30b-a3b-instruct-2507"
# DEFAULT_EXTRACTOR_MODEL_ID = "deepseek/deepseek-v3.2:nitro"
DEFAULT_EXTRACTOR_MODEL_ID = "openai/gpt-5.5"
DEFAULT_CONCURRENCY = 10


def _extractor_cfg() -> dict:
    return get_cfg().get("tools_config", {}).get("eci_extractor", {})


# =============================================================================
# UTILS
# =============================================================================

@lru_cache(maxsize=1)
def _make_llm() -> ChatOpenAI:
    model_cfg = get_cfg().get("model", {})
    extractor_cfg = _extractor_cfg()
    return ChatOpenAI(
        model=extractor_cfg.get("model_id", DEFAULT_EXTRACTOR_MODEL_ID),
        temperature=0.0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=model_cfg.get("base_url", "https://openrouter.ai/api/v1"),
    )


register_reset(_make_llm.cache_clear)

SUB_CALL_SYSTEM_PROMPT = (
    "You are an Event Causality Identification (ECI) specialist.\n"
    "You are given ONE target event mention and a list of candidate event\n"
    "mentions, all tagged <ID text> in the document. Use the exact IDs given\n"
    "— never invent, renumber, or paraphrase one.\n\n"
    "For EACH candidate, decide the causal relation to the target:\n"
    "  - \"causes\":    the TARGET causes/enables/leads to the candidate.\n"
    "  - \"causedby\":  the candidate causes/enables/leads to the TARGET.\n"
    "  - \"norel\":     no causal link either way.\n\n"
    "Two examples of common mistakes to avoid:\n\n"
    "NOT CAUSAL (commonly mislabeled as causal): \"The massacre contributed to\n"
    "calls for reprisals, leading to the 1779 Sullivan Expedition which drove\n"
    "the Iroquois out of western New York.\" Here 'contributed' and 'drove' are\n"
    "NOT directly causally linked — they are two separate links in the same\n"
    "chain (massacre -> reprisals -> Expedition -> drove out), each pointing to\n"
    "the Expedition, not to each other. Do not flatten a multi-step chain into\n"
    "a direct link between two events that are each only adjacent to a third.\n\n"
    "CAUSAL (commonly missed): \"...the air strikes flown during deny flight\n"
    "led to operation deliberate force, a massive nato bombing campaign in\n"
    "bosnia that played a key role in ending the war.\" Here 'operation'\n"
    "(deliberate force) DOES cause 'ending' (the war) — phrases like 'played a\n"
    "key role in' are just as causal as 'led to' or 'caused', and a generic\n"
    "trigger word like 'operation' can still be the causal subject.\n\n"
    "Output ONLY a JSON array, one entry per candidate, in the form\n"
    "{\"id\": \"<candidate id>\", \"relation\": \"causes\"|\"causedby\"|\"norel\"}.\n"
    "No markdown fences, no commentary, no reasoning fields."
)


def _sub_call_user_prompt(document_text: str, target: Dict[str, str], candidates: List[Dict[str, str]]) -> str:
    candidate_lines = "\n".join(f'  - {c["id"]} ("{c.get("text", "?")}")' for c in candidates)
    return f"""Document:
{document_text}

TARGET mention: {target["id"]} ("{target.get("text", "?")}")

Candidate mentions:
{candidate_lines}

For each candidate, output its relation to the target."""


async def _evaluate_target(
    llm: ChatOpenAI, document_text: str, target: Dict[str, str], candidates: List[Dict[str, str]]
) -> List[Tuple[str, str]]:
    """Returns a list of (causer_id, effect_id) facts found for this target, one sub-call."""
    if not candidates:
        return []

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SUB_CALL_SYSTEM_PROMPT),
            HumanMessage(content=_sub_call_user_prompt(document_text, target, candidates)),
        ])
        # Logged the same way the orchestrator's own calls are (one line per
        # call, appended to the run's shared trace file) so this sub-call's
        # tokens/cost are picked up by the existing trace-based accounting in
        # mlflow_tracker.py instead of being invisible to it.
        trace_dump.trace_dump({"messages": [response]})
        raw = response.content
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        json_str = match.group(0) if match else raw
        judgements = json.loads(json_str)
    except Exception:
        return []

    facts: List[Tuple[str, str]] = []
    target_id = target["id"]
    for j in judgements:
        candidate_id = j.get("id")
        relation = j.get("relation")
        if not candidate_id or relation not in ("causes", "causedby"):
            continue
        if relation == "causes":
            facts.append((target_id, candidate_id))
        else:
            facts.append((candidate_id, target_id))
    return facts


# =============================================================================
# TOOL
# =============================================================================

@tool
async def eci_extractor(document_text: str, mentions: List[Dict[str, str]]) -> str:
    """
    Extracts ALL causal relations in a document. Call this tool exactly ONCE
    per document, passing every event mention — internally it evaluates each
    mention against all others concurrently (on a separate, smaller model)
    and aggregates the results itself, so none of that intermediate work
    enters your context.

    Its returned JSON array IS the final answer: already deduplicated,
    already bidirectional (causes + causedby). Forward it as-is rather than
    re-deriving or rewriting it.

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

    llm = _make_llm()
    concurrency = _extractor_cfg().get("concurrency", DEFAULT_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_evaluate(target: Dict[str, str], candidates: List[Dict[str, str]]):
        async with semaphore:
            return await _evaluate_target(llm, document_text, target, candidates)

    tasks = [
        _bounded_evaluate(target, [m for m in mentions if m["id"] != target["id"]])
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
