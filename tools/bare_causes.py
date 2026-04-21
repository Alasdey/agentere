# tools/bare_causes.py
"""
Tool that distils a document into its bare causal semantics.

A complex narrative is reduced to a small set of core events and
the causal relations that connect them, stripping away all narrative
detail, attribution, and temporal filler.
"""

from __future__ import annotations

import json
import os
import re
import yaml
from functools import lru_cache
from typing import Any, Dict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =============================================================================
# UTILS
# =============================================================================

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)


def _get_config() -> Dict[str, Any]:
    return _CFG


@lru_cache(maxsize=1)
def _make_llm() -> ChatOpenAI:
    model_cfg = _CFG.get("model", {})
    return ChatOpenAI(
        model=model_cfg.get("default_model_id", "openai/gpt-4o-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=model_cfg.get("base_url", "https://openrouter.ai/api/v1"),
    )


# =============================================================================
# TOOL
# =============================================================================

@tool
async def bare_causes(context_text: str) -> str:
    """
    Extracts the bare causal semantics from a document.

    Use this tool to reduce a complex narrative to its core causal skeleton:
    a minimal set of abstract events and the causal links between them.
    All narrative detail, attribution, hedging, and temporal filler is stripped.
    The output helps you reason about which event pairs are causally connected
    before committing to specific relation labels.

    Args:
        context_text: The full document text to distil.

    Returns:
        A JSON string containing:
        - "events": a list of {"id": str, "description": str} with short
          abstract descriptions of the core events.
        - "causal_links": a list of {"cause": event_id, "effect": event_id,
          "basis": str} capturing why the link holds.
        - "summary": a one-sentence plain-language summary of the causal
          chain in the document.
    """

    llm = _make_llm()

    system_prompt = (
        "You are a causal-semantics analyst. Your task is to read a document "
        "and extract ONLY the bare causal skeleton: the minimal set of core "
        "events and the causal relations between them.\n\n"
        "Rules:\n"
        "1. Identify the FEWEST events necessary to capture all causal "
        "information in the text (typically 3-8 events).\n"
        "2. Describe each event in ≤10 words, abstracting away specific "
        "names, dates, and counts unless they are causally relevant.\n"
        "3. For every causal link, state which event causes which and give a "
        "≤15-word basis grounded in the text.\n"
        "4. Do NOT invent causal links that are not supported by the text.\n"
        "5. Do NOT include temporal-only relations (e.g., 'A happened before B' "
        "without causation).\n"
        "6. Output ONLY valid JSON — no commentary, no markdown fences."
    )

    user_prompt = f"""Document:
{context_text}

Extract the bare causal skeleton. Output a JSON object with exactly three keys:

{{
  "events": [
    {{"id": "E1", "description": "..."}},
    {{"id": "E2", "description": "..."}}
  ],
  "causal_links": [
    {{"cause": "E1", "effect": "E2", "basis": "..."}}
  ],
  "summary": "One sentence describing the overall causal chain."
}}"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content

        # Extract JSON from potential markdown fences or conversational filler
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        json_str = match.group(0) if match else raw

        # Validate and re-serialize
        data = json.loads(json_str)
        return json.dumps(data, indent=2)

    except json.JSONDecodeError as e:
        return f"Error: LLM returned invalid JSON — {e}"
    except Exception as e:
        return f"Error executing bare_causes analysis: {e}"