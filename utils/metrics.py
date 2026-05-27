# FILE: utils/metrics.py
from __future__ import annotations
from typing import List, Dict, Any, Set

from utils.labels import NOREL_VARIANTS as _NOREL_NORM  # re-export for resample.py import

def compute_ere_metrics(gold_triples: List[tuple], pred_triples: List[tuple]):
    """
    Computes set-based precision, recall, and F1 for relation triples.
    Expected format: (src_id, label, tgt_id)
    Useful for simple checks, but not for the full log report.
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
        "tp": tp
    }

def compute_multiclass_metrics(y_true: List[str], y_pred: List[str], labels: List[str]):
    """
    Standard multiclass classification metrics (Precision/Recall/F1 per label).
    Replicates the logic from the old code.
    """
    # Normalize inputs to ensure strings
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]

    # Initialize counts
    # If dynamic labels in data exceed provided labels list, add them or they won't be counted in PER-LABEL
    # But usually we want strict adherence to the schema.
    
    unique_labels = set(labels) | set(y_true) | set(y_pred)
    # Ensure NoRel is tracked but maybe excluded from macro calculation depending on preference
    
    counts = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in unique_labels}
    
    for gt, pr in zip(y_true, y_pred):
        if gt == pr:
            counts[gt]["tp"] += 1
        else:
            counts[gt]["fn"] += 1
            counts[pr]["fp"] += 1

    per_label = {}
    for lab in unique_labels:
        tp, fp, fn = counts[lab]["tp"], counts[lab]["fp"], counts[lab]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    # Macro: usually exclude NoRel and heavily skewed classes if strictly following strict eval, 
    # but here we follow standard logic: avg of all relevant classes.
    eval_labels = [lab for lab in unique_labels if lab.lower() not in _NOREL_NORM]
    
    if not eval_labels:
        macro_f1 = 0.0
    else:
        macro_f1 = sum(per_label[lab]["f1"] for lab in eval_labels) / len(eval_labels)

    # Micro: Aggregate TP/FP/FN across positive classes
    total_tp = sum(counts[lab]["tp"] for lab in eval_labels)
    total_fp = sum(counts[lab]["fp"] for lab in eval_labels)
    total_fn = sum(counts[lab]["fn"] for lab in eval_labels)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "per_label": per_label,
        "macro_f1": macro_f1,
        "micro_precision": micro_prec, 
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "total": len(y_true)
    }

def compute_binary_metrics(y_true: List[str], y_pred: List[str]):
    """
    Computes metrics treating ANY relation as Positive versus NoRel as Negative.
    """
    def binarize(y):
        return "NEG" if str(y).lower() in _NOREL_NORM else "POS"
        
    yt = [binarize(x) for x in y_true]
    yp = [binarize(x) for x in y_pred]
    
    tp = sum(1 for a, b in zip(yt, yp) if a == "POS" and b == "POS")
    fp = sum(1 for a, b in zip(yt, yp) if a == "NEG" and b == "POS")
    fn = sum(1 for a, b in zip(yt, yp) if a == "POS" and b == "NEG")
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        "precision": prec, 
        "recall": rec, 
        "f1": f1, 
        "support_pos": sum(1 for v in yt if v == "POS")
    }