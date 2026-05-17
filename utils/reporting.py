# FILE: utils/reporting.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from utils.metrics import compute_multiclass_metrics, compute_binary_metrics

def reconstruct_pairwise_predictions(
    doc_id: str,
    doc_idx: int,
    gold_triples: List[Tuple[str, str, str]],
    pred_triples: List[Tuple[str, str, str]],
    pair_stats: Dict[str, Any],
    lang: str = "unknown"
) -> Dict[str, Any]:
    """
    Reconstructs the full list of pair predictions (Gold vs Pred).
    Embeds 'vote_counts' from resampling stats directly into the rows.
    """
    gold_map = {(src, tgt): lbl for src, lbl, tgt in gold_triples}
    pred_map = {(src, tgt): lbl for src, lbl, tgt in pred_triples}
    
    # Union of all relevant pairs (Gold + Pred)
    all_pairs = set(gold_map.keys()) | set(pred_map.keys())

    pairwise_rows = []
    y_true_doc = []
    y_pred_doc = []

    for src, tgt in all_pairs:
        gold_lbl = gold_map.get((src, tgt), "NoRel")
        pred_lbl = pred_map.get((src, tgt), "NoRel")
        
        y_true_doc.append(gold_lbl)
        y_pred_doc.append(pred_lbl)

        pair_key = f"{src},{tgt}"
        
        # Retrieve stats if available (votes, counts)
        stats = pair_stats.get(pair_key, {})

        pairwise_rows.append({
            "doc_idx": doc_idx,
            "id": doc_id,
            "lang": lang,
            "pair": pair_key,
            "gold": gold_lbl,
            "pred": pred_lbl,
            "vote_counts": stats.get("vote_counts", None),
        })

    return {
        "rows": pairwise_rows,
        "y_true": y_true_doc,
        "y_pred": y_pred_doc
    }


def generate_run_report(
    results: List[Dict[str, Any]], 
    total_processed_count: int,
    config: Dict[str, Any],
    valid_labels: List[str] = None
) -> Dict[str, Any]:
    """
    Aggregates individual document results into a global report.
    Calculates Global, Per-Doc, and Per-Language metrics.
    Formulates the final structure for the logger.
    """
    if valid_labels is None:
        raise ValueError("valid_labels must be provided — pass ds_config['labels'] from config")

    # Containers
    labels_all_true = []
    labels_all_pred = []
    
    per_doc_metrics_log = []
    per_pair_predictions_log = []
    per_lang_stats = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    valid_results = [r for r in results if r is not None]
    retry_failed_count = sum(1 for r in valid_results if r.get("context", {}).get("retry_failure"))

    # --- 1. Iterate over docs ---
    for res in valid_results:
        # A. Pairwise Reconstruction (Includes Vote Counts)
        alignment = reconstruct_pairwise_predictions(
            doc_id=res["id"],
            doc_idx=res["doc_idx"],
            gold_triples=res["gold_triples"],
            pred_triples=res["pred_triples"],
            pair_stats=res.get("pair_stats", {}),
            lang=res["lang"]
        )
        
        y_true = alignment["y_true"]
        y_pred = alignment["y_pred"]
        
        # Accumulate Global
        labels_all_true.extend(y_true)
        labels_all_pred.extend(y_pred)
        
        # Accumulate Detailed Logs
        per_pair_predictions_log.extend(alignment["rows"])
        
        # Accumulate Per-Lang
        per_lang_stats[res["lang"]]["y_true"].extend(y_true)
        per_lang_stats[res["lang"]]["y_pred"].extend(y_pred)

        # B. Per-Doc Metrics
        mc_d = compute_multiclass_metrics(y_true, y_pred, valid_labels)
        binm_d = compute_binary_metrics(y_true, y_pred)
        
        per_doc_metrics_log.append({
            "doc_idx": res["doc_idx"],
            "id": res["id"],
            "pairs": len(y_true),
            "macro_f1": mc_d["macro_f1"],
            "micro_f1": mc_d["micro_f1"],
            "micro_precision": mc_d["micro_precision"],
            "micro_recall": mc_d["micro_recall"],
            "per_label": mc_d["per_label"],
            "binary": binm_d,
        })

    # --- 2. Compute Global Metrics ---
    # Note: Only compute if we have results, otherwise 0.0 defaults within metrics utils
    global_mc = compute_multiclass_metrics(labels_all_true, labels_all_pred, valid_labels)
    global_bin = compute_binary_metrics(labels_all_true, labels_all_pred)
    
    # --- 3. Compute Per-Lang Metrics ---
    per_lang_metrics_log = {}
    for lg, data in per_lang_stats.items():
        if not data["y_true"]: continue
        per_lang_metrics_log[lg] = {
            "multiclass": compute_multiclass_metrics(data["y_true"], data["y_pred"], valid_labels),
            "binary": compute_binary_metrics(data["y_true"], data["y_pred"]),
            "total_pairs": len(data["y_true"])
        }

    # --- 4. Construct Final Dictionary ---
    final_report = {
        # Top-level Global Metrics
        "per_label": global_mc["per_label"],
        "macro_f1": global_mc["macro_f1"],
        "micro_precision": global_mc["micro_precision"],
        "micro_recall": global_mc["micro_recall"],
        "micro_f1": global_mc["micro_f1"],
        "total_pairs": global_mc["total"],
        "binary": global_bin,
        
        # Metadata
        "skipped_docs": total_processed_count - len(valid_results),
        "retry_failed_docs": retry_failed_count,
        
        # Detailed Lists
        "per_doc_metrics": per_doc_metrics_log,
        "per_lang_metrics": per_lang_metrics_log,
        "per_pair_predictions": per_pair_predictions_log, 
        
        # Config Snapshots
        "resampling": {
            "enabled": config["experiment"]["resampling"]["enabled"],
            "n_runs": config["experiment"]["resampling"]["n_runs"],
            "tie_breaking": config["experiment"]["resampling"]["tie_breaking"],
        }
    }
    
    return final_report