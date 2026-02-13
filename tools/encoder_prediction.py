# tools/encoder_predictions.py
"""
Tool that exposes encoder-baseline predictions to the LLM agent.

The predictions JSON is loaded once and cached. The LLM calls this tool
with only a document ID; everything else is resolved internally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.tools import tool

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHE  (loaded once, reused across all tool calls)
# ═══════════════════════════════════════════════════════════════════════════════

_CACHE: Optional[Dict[str, Any]] = None
_CACHE_PATH: Optional[str] = None


def _get_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_predictions() -> Dict[str, Any]:
    """
    Loads the encoder predictions JSON once and caches it.
    Expects config.yaml to contain:

        encoder_predictions:
          path: "encoder_baseline/logs/<run>/predictions.json"
    """
    global _CACHE, _CACHE_PATH

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
    _CACHE_PATH = pred_path

    n_samples = len(_CACHE.get("samples", {}))
    print(
        f"[encoder_predictions] Loaded {n_samples} documents "
        f"from {pred_path}"
    )
    return _CACHE


def _format_predictions(doc_id: str, doc_data: List[Dict]) -> str:
    """Formats a single document's predictions into a readable string."""
    lines = [f"Encoder classifier predictions for document \"{doc_id}\":\n"]

    for pair in doc_data:
        src = pair.get("src", "?")
        src_text = pair.get("src_text", "")
        tgt = pair.get("tgt", "?")
        tgt_text = pair.get("tgt_text", "")

        pred_label = pair.get("pred_label", "NoRel")
        pred_probs = pair.get("pred_prob", [])
        gold_label = pair.get("gold_label", None)

        # Find the confidence of the predicted label
        confidence = max(pred_probs) if pred_probs else 0.0

        line = (
            f'  {src} ("{src_text}") → {tgt} ("{tgt_text}"): '
            f"{pred_label} (confidence: {confidence:.2f})"
        )
        lines.append(line)

    if len(doc_data) == 0:
        lines.append("  (no pairs)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def encoder_predictions(doc_id: str) -> str:
    """
    Retrieves the encoder-based classifier's relation predictions for a document.

    Use this tool to get a second opinion from a trained neural classifier
    before making your final relation predictions. Call it with the document ID
    shown in the prompt.

    Args:
        doc_id: The document identifier (e.g. "causal-en_15_WSJ_4.ann").

    Returns:
        A summary of predicted relations with confidence scores,
        or an error message if the document is not found.
    """
    try:
        data = _load_predictions()
    except FileNotFoundError as e:
        return f"Error: {e}"

    samples = data.get("samples", {})
    label_list = data.get("label_list", [])
    threshold = data.get("threshold", 0.5)

    if doc_id not in samples:
        # Try partial match (some IDs may have extensions stripped)
        candidates = [k for k in samples if doc_id in k or k in doc_id]
        if len(candidates) == 1:
            doc_id = candidates[0]
        elif len(candidates) > 1:
            return (
                f"Ambiguous doc_id '{doc_id}'. "
                f"Candidates: {candidates[:5]}"
            )
        else:
            return (
                f"Document '{doc_id}' not found in encoder predictions. "
                f"Available: {list(samples.keys())[:10]}..."
            )

    doc_pairs = samples[doc_id]
    summary = _format_predictions(doc_id, doc_pairs)

    meta = (
        f"\n\n[Model: {data.get('config', {}).get('model', '?')}, "
        f"threshold: {threshold:.3f}, "
        f"labels: {label_list}]"
    )

    return summary + meta