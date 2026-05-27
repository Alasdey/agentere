# tools/encoder.py
"""
Tool that exposes encoder-baseline predictions to the LLM agent.
The current document ID is read from a contextvars.ContextVar,
set by the orchestrator before each graph invocation.
The LLM calls this tool with no arguments.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.tools import tool

from utils.context import CURRENT_DOC_ID

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHE  (loaded once, reused across all calls)
# ═══════════════════════════════════════════════════════════════════════════════

_CACHE: Optional[Dict[str, Any]] = None


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CFG = yaml.safe_load(_f)


def _load_predictions() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    config = _CFG
    pred_path = config.get("encoder", {}).get("path", "")

    if not pred_path or not os.path.isfile(pred_path):
        raise FileNotFoundError(
            f"Encoder predictions file not found at '{pred_path}'. "
            f"Set 'encoder.path' in config.yaml."
        )

    with open(pred_path, "r", encoding="utf-8") as f:
        _CACHE = json.load(f)

    n = len(_CACHE.get("samples", {}))
    # print(f"[encoder] Loaded {n} documents from {pred_path}")
    return _CACHE


# ═══════════════════════════════════════════════════════════════════════════════
# NoRel FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_norel(
    doc_data: List[Dict],
    label_list: List[str],
    threshold: float,
    filter_cfg: Dict[str, Any],
) -> List[Dict]:
    """
    Removes NoRel predictions whose highest relation probability is more
    than *delta* below the decision threshold.

    For a pair classified as NoRel, let
        max_rel_prob = max(pred_prob)          # best relation score
    The pair is **kept** when  max_rel_prob ≥ threshold − delta  (uncertain)
    and **dropped** otherwise (encoder is very confident there is no relation).

    Non-NoRel predictions are always kept.

    Args:
        doc_data:    List of pair dicts for one document.
        label_list:  Ordered label names matching pred_prob indices.
        threshold:   Decision threshold from the encoder config.
        filter_cfg:  The ``encoder.filter_norel`` sub-dict.

    Returns:
        Filtered list of pair dicts.
    """
    if not filter_cfg.get("enabled", False):
        return doc_data

    delta = float(filter_cfg.get("delta", 0.1))
    cutoff = threshold - delta          # pairs below this are discarded

    kept: List[Dict] = []
    for pair in doc_data:
        pred_label = pair.get("pred_label", "NoRel")

        # Always keep pairs that have an actual relation prediction
        if pred_label != "NoRel":
            kept.append(pair)
            continue

        # For NoRel: check whether encoder was *uncertain*
        pred_probs = pair.get("pred_prob", [])
        if not pred_probs:
            # No probability info → keep to be safe
            kept.append(pair)
            continue

        max_rel_prob = max(pred_probs)

        if max_rel_prob >= cutoff:
            # Close to the boundary → the LLM should see this pair
            kept.append(pair)
        # else: far below threshold → drop silently

    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def _format_predictions(
    doc_id: str,
    doc_data: List[Dict],
    label_list: list,
    total_before_filter: int,
) -> str:
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

    # Inform the LLM how many low-confidence NoRel pairs were omitted
    n_dropped = total_before_filter - len(doc_data)
    if n_dropped > 0:
        lines.append(
            f"\n  [{n_dropped} highly-confident NoRel pairs omitted "
            f"(far below threshold)]"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL  (no arguments — doc_id comes from context)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
async def encoder(comment: str="") -> str:
    """
    Retrieves the encoder-based classifier's relation predictions
    for the current document. Call this tool with no arguments to get
    a second opinion from a trained neural model before making your
    final predictions.

    Returns:
        A summary listing each mention pair with the encoder's predicted
        label and per-class confidence scores.
    """
    # comment parm satisfies the schema

    doc_id = CURRENT_DOC_ID.get()

    if not doc_id:
        print(f"encoder: no {doc_id}")
        return "Error: No document context available."

    try:
        if _CACHE is None:
            data = await asyncio.to_thread(_load_predictions)
        else:
            data = _CACHE
    except FileNotFoundError as e:
        return f"Error: {e}"

    samples = data.get("samples", {})
    label_list = data.get("label_list", [])
    threshold = data.get("threshold", 0.5)

    # Exact match first, then fuzzy
    resolved_id = doc_id
    if doc_id not in samples:
        candidates = [k for k in samples if doc_id in k or k in doc_id]
        if len(candidates) == 1:
            resolved_id = candidates[0]
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

    doc_data = samples[resolved_id]
    total_before_filter = len(doc_data)

    # ── Apply NoRel filter ──────────────────────────────────────────────
    config = _CFG
    filter_cfg = config.get("encoder", {}).get("filter_norel", {})
    doc_data = _filter_norel(doc_data, label_list, threshold, filter_cfg)

    summary = _format_predictions(
        doc_id, doc_data, label_list, total_before_filter
    )

    meta = (
        f"\n\n[Model: {data.get('config', {}).get('model', '?')}, "
        f"threshold: {threshold:.3f}, labels: {label_list}, "
        f"pairs shown: {len(doc_data)}/{total_before_filter}]"
    )
    return summary + meta