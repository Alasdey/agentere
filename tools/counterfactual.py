# tools/counterfactual.py
from __future__ import annotations

import os
import json
import re
import yaml
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =============================================================================
# UTILS & CONFIG
# =============================================================================

def _get_config() -> Dict[str, Any]:
    # Paths relative to the tool file in the /tools/ directory
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
def counterfactual_test(
    pair: str,
    context_text: str,
    event_i_text: Optional[str] = None,
    event_j_text: Optional[str] = None,
) -> str:
    """
    Performs a 'but-for' counterfactual analysis for a pair of event triggers.
    This tool answers if one event is a necessary precondition or cause for another.

    Args:
        pair: The event IDs being tested, e.g., "e1,e2" or "T1,T2".
        context_text: The document text provided in the prompt.
        event_i_text: The specific text/trigger span for the first event.
        event_j_text: The specific text/trigger span for the second event.

    Returns:
        A JSON string containing the 'but-for' results for both directions and reasoning.
    """
    if "," not in pair:
        return "Error: Pair parameter must be 'ID1,ID2'."

    ei, ej = [p.strip() for p in pair.split(",", 1)]
    llm = _make_llm()

    sys_prompt = (
        "You are a logical reasoning assistant. Your task is to perform 'but-for' counterfactual "
        "tests on event pairs. Do not provide labels like 'Cause' or 'NoRel'. Provide ONLY the "
        "factual necessity found in the text."
    )

    user_prompt = f"""
Context:
{context_text}

Analyze the dependency between:
Event A: {ei} (Text: "{event_i_text or 'Refer to context'}")
Event B: {ej} (Text: "{event_j_text or 'Refer to context'}")

Tasks:
1. Test (B without A): If Event A had not occurred, would Event B still have happened?
2. Test (A without B): If Event B had not occurred, would Event A still have happened?

Output ONLY a JSON object:
{{
  "pair": "{pair}",
  "reasoning": "brief semantic analysis",
  "b_is_dependent_on_a": "yes/no/unclear",
  "a_is_dependent_on_b": "yes/no/unclear"
}}
"""

    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        raw_content = response.content
        
        # Clean potential markdown JSON fences
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        json_str = match.group(0) if match else raw_content
        
        # We parse and re-dump to ensure it is valid compressed JSON for the model
        data = json.loads(json_str)
        return json.dumps(data)
    
    except Exception as e:
        return f"Error executing counterfactual analysis: {str(e)}"