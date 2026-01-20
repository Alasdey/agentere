# FILE: utils/metrics.py
from __future__ import annotations
from typing import List, Dict, Any, Set

def compute_ere_metrics(gold_triples: List[tuple], pred_triples: List[tuple]):
    """
    Computes precision, recall, and F1 for relation triples.
    Expected format: (src_id, label, tgt_id)
    """
    gold_set = set(gold_triples)
    pred_set = set(pred_triples)

    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(gold_set),
        "predicted": len(pred_set),
        "tp": tp,
        "Causal existence": None,
    }