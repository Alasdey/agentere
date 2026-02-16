# tools/eci.py
"""
Tool inspired by the HOTECI framework (Man et al., 2024) for
Event Causality Identification.

Instead of testing one pair at a time (quadratic in the number of
mentions), this tool takes a SINGLE target mention and identifies
ALL other mentions in the document that are causally related to it.
This makes the number of required tool calls linear in the number
of mentions rather than quadratic in the number of pairs.

Mirrors the HOTECI philosophy:
  - Hierarchical context selection  (sentences → words)
  - Binary causal question  L(e_s, e_t) ∈ {Yes, No}
  - Output includes salient context grounding the decision
"""

from __future__ import annotations

import json
import os
import re
import yaml
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =============================================================================
# UTILS
# =============================================================================

def _get_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_llm() -> ChatOpenAI:
    cfg = _get_config()
    model_cfg = cfg.get("model", {})
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
def eci(
    target_mention: str,
    target_text: str,
    all_mentions: List[Dict[str, str]],
    context_text: str,
) -> str:
    """
    Identifies all event mentions causally related to a target mention.

    Inspired by the HOTECI generative ECI framework, this tool answers:
    "Which of the other mentions in this document are causally related
    to the target mention?" — regardless of direction or specific type.

    It also returns the key context sentences and words that ground each
    detected causal link, following the HOTECI principle that selecting
    important context improves causal judgement.

    Call this tool ONCE PER MENTION (linear complexity) instead of once
    per pair (quadratic complexity).

    Args:
        target_mention: The ID of the target event mention (e.g. "T3").
        target_text: The trigger/span text of the target mention (e.g. "earthquake").
        all_mentions: List of ALL other mentions in the document, each a dict
                      with keys "id" and "text".
                      Example: [{"id": "T0", "text": "collapse"}, {"id": "T1", "text": "flood"}]
        context_text: The full document text.

    Returns:
        A JSON string containing:
        - "target": the target mention ID
        - "causal_mentions": list of objects for each mention found to be
          causally related, each with:
            - "id": the related mention ID
            - "text": its trigger text
            - "confidence": "high" / "medium" / "low"
            - "important_words": salient bridging words
            - "reasoning": ≤1-sentence explanation
        - "non_causal_mentions": list of mention IDs with no causal link
        - "important_sentences": the 1-4 most relevant sentences selected
          from the document for the target mention
    """

    if not all_mentions:
        return json.dumps({
            "target": target_mention,
            "causal_mentions": [],
            "non_causal_mentions": [],
            "important_sentences": [],
            "note": "No other mentions provided."
        })

    llm = _make_llm()

    # Format the mention list for the prompt
    mention_lines = "\n".join(
        f'  - {m["id"]} ("{m.get("text", "?")}")'
        for m in all_mentions
    )

    # ------------------------------------------------------------------
    # System prompt mirrors the HOTECI 3-step design:
    #   Step 1: sentence-level context selection  (≈ sentence OT)
    #   Step 2: per-candidate word-level evidence  (≈ word OT)
    #   Step 3: binary causal judgement per candidate
    # ------------------------------------------------------------------
    system_prompt = (
        "You are an Event Causality Identification (ECI) specialist.\n"
        "Your task has THREE ordered steps — follow them exactly:\n\n"
        "STEP 1 — Context Selection (sentence level):\n"
        "  Read the full document and identify the 1-4 sentences most\n"
        "  relevant to the TARGET mention's causal role. Prefer sentences\n"
        "  semantically close to the target trigger and containing causal\n"
        "  cues (e.g., 'caused', 'led to', 'because', 'result', 'due to').\n\n"
        "STEP 2 — Per-candidate evidence (word level):\n"
        "  For EACH candidate mention, determine whether causal evidence\n"
        "  connects it to the target. Extract the specific bridging words\n"
        "  or phrases that support or refute a causal link.\n\n"
        "STEP 3 — Binary Causal Judgement:\n"
        "  For each candidate, answer: 'Is there a causal relation between\n"
        "  the target and this candidate?' (Yes / No).\n"
        "  Do NOT determine direction or specific relation type.\n\n"
        "Output ONLY a valid JSON object — no markdown fences, no commentary."
    )

    user_prompt = f"""Document:
{context_text}

TARGET mention: {target_mention} ("{target_text}")

Candidate mentions:
{mention_lines}

For EACH candidate, determine whether there is a causal relation with the
TARGET (in either direction). Output JSON:

{{
  "target": "{target_mention}",
  "important_sentences": ["sentence1", "sentence2"],
  "causal_mentions": [
    {{
      "id": "mention_id",
      "text": "trigger text",
      "confidence": "high / medium / low",
      "important_words": ["word1", "word2"],
      "reasoning": "brief explanation"
    }}
  ],
  "non_causal_mentions": ["id1", "id2"]
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content

        # Extract JSON robustly
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        json_str = match.group(0) if match else raw

        data = json.loads(json_str)
        data.setdefault("target", target_mention)

        return json.dumps(data)

    except json.JSONDecodeError as e:
        return json.dumps({
            "target": target_mention,
            "error": f"LLM returned invalid JSON: {e}",
            "raw": raw[:500] if "raw" in dir() else "",
        })
    except Exception as e:
        return json.dumps({
            "target": target_mention,
            "error": f"ECI analysis failed: {e}",
        })