# tools/pair_extractor.py
from __future__ import annotations

import os
import yaml
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from tools.few_shot import CURRENT_DOC_TEXT

CURRENT_MENTIONS_MAP: ContextVar[dict] = ContextVar("current_mentions_map", default={})

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)


@lru_cache(maxsize=1)
def _make_llm() -> ChatOpenAI:
    cfg = _CFG.get("pair_extractor", {})
    model_id = cfg.get("model_id") or _CFG["model"]["default_model_id"]
    base_url = _CFG["model"].get("base_url", "https://openrouter.ai/api/v1")
    return ChatOpenAI(
        model=model_id,
        temperature=0.0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=base_url,
    )


_SYSTEM = (
    "You are an expert in event causality. "
    "Given a document and a list of event mentions, identify ALL pairs of events "
    "that are potentially in a causal or enabling relation. "
    "Prioritise RECALL — it is better to include a borderline pair than to miss a true relation. "
    "Do NOT assign labels; just list candidate pairs."
)


@tool
async def pair_extractor(comment: str = "") -> str:
    """
    Uses a secondary LLM to scan the document and enumerate every pair of event mentions
    that could plausibly be in a causal relation (PRECONDITION / FALLING_ACTION / CAUSE …).
    Returns a plain-text list of candidate pairs — one per line — with mention IDs and texts.
    Call this when you want an exhaustive candidate set before labelling, to avoid missing pairs.
    """
    doc_text = CURRENT_DOC_TEXT.get()
    mentions_map: dict = CURRENT_MENTIONS_MAP.get()

    if not mentions_map:
        return "No mention map available — cannot extract candidate pairs."

    mention_lines = "\n".join(
        f"  {mid}: \"{text}\"" for mid, text in mentions_map.items()
    )

    user_prompt = (
        f"Document:\n{doc_text}\n\n"
        f"Event mentions:\n{mention_lines}\n\n"
        "List every pair (ID_i, ID_j) where the events might be causally related "
        "(either direction). One pair per line, format: ID_i, ID_j — reason.\n"
        "Include ALL potentially positive pairs. Omit only pairs that are clearly unrelated."
    )

    llm = _make_llm()
    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()
