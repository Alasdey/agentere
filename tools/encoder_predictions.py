# tools/encoder_predictions.py
"""
Tool that exposes encoder-baseline predictions to the LLM agent.
The current document ID is read from a contextvars.ContextVar,
set by the orchestrator before each graph invocation.
The LLM calls this tool with no arguments.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.tools import tool

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT VARIABLE — set by main.py, read by the tool
# ═══════════════════════════════════════════════════════════════════════════════

CURRENT_DOC_ID: ContextVar[str] = ContextVar("current_doc_id", default="")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHE  (loaded once, reused across all calls)
# ═══════════════════════════════════════════════════════════════════════════════

_CACHE: Optional[Dict[str, Any]] = None


def _get_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_predictions() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    config = _get_config()
    pred_path = config.get("encoder_predictions", {}).get("path", "")

    if not pred_path or not os.path.isfile(pred_path):
        raise FileNotFoundError(
            f"Encoder predictions file not found at '{pred_path}'. "
            f"Set 'encoder_predictions.path' in config.yaml."
        )

    with open(pred_path, "r", encoding="utf-8") as f:
        _CACHE = json.load(f)

    n = len(_CACHE.get("samples", {}))
    print(f"[encoder_predictions] Loaded {n} documents from {pred_path}")
    return _CACHE


def _format_predictions(doc_id: str, doc_data: List[Dict], label_list: list) -> str:
    lines = [f"Encoder classifier predictions for document \"{doc_id}\":\n"]

    for pair in doc_data:
        src = pair.get("src", "?")
        src_text = pair.get("src_text", "")
        tgt = pair.get("tgt", "?")
        tgt_text = pair.get("tgt_text", "")
        pred_label = pair.get("pred_label", "NoRel")
        pred_probs = pair.get("pred_prob", [])

        # Per-label confidences
        if pred_probs and label_list and len(pred_probs) == len(label_list):
            details = ", ".join(
                f"{lab}: {p:.2f}" for lab, p in zip(label_list, pred_probs)
            )
            conf_str = f"[{details}]"
        else:
            conf_str = f"(confidence: {max(pred_probs) if pred_probs else 0:.2f})"

        lines.append(
            f'  {src} ("{src_text}") → {tgt} ("{tgt_text}"): '
            f"{pred_label} {conf_str}"
        )

    if not doc_data:
        lines.append("  (no pairs)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL  (no arguments — doc_id comes from context)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def encoder_predictions() -> str:
    """
    Retrieves the encoder-based classifier's relation predictions
    for the current document. Call this tool with no arguments to get
    a second opinion from a trained neural model before making your
    final predictions.

    Returns:
        A summary listing each mention pair with the encoder's predicted
        label and per-class confidence scores.
    """

    doc_id = CURRENT_DOC_ID.get()

    if not doc_id:
        print(f"encoder_predictions: no {doc_id}")
        return "Error: No document context available."

    try:
        data = _load_predictions()
    except FileNotFoundError as e:
        return f"Error: {e}"

    samples = data.get("samples", {})
    label_list = data.get("label_list", [])
    threshold = data.get("threshold", 0.5)

    # Exact match first, then fuzzy
    if doc_id not in samples:
        candidates = [k for k in samples if doc_id in k or k in doc_id]
        if len(candidates) == 1:
            doc_id = candidates[0]
        elif candidates:
            return (
                f"Ambiguous doc_id '{doc_id}'. "
                f"Candidates: {candidates[:5]}"
            )
        else:
            return (
                f"Document '{doc_id}' not found in encoder predictions. "
                f"Available: {list(samples.keys())[:10]}..."
            )

    summary = _format_predictions(doc_id, samples[doc_id], label_list)

    meta = (
        f"\n\n[Model: {data.get('config', {}).get('model', '?')}, "
        f"threshold: {threshold:.3f}, labels: {label_list}]"
    )
    return summary + meta