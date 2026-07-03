# FILE: utils/reporting.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from utils.labels import NOREL
from utils.metrics import compute_multiclass_metrics, compute_binary_metrics

def reconstruct_pairwise_predictions(
    doc_id: str,
    doc_idx: int,
    gold_triples: List[Tuple[str, str, str]],
    pred_triples: List[Tuple[str, str, str]],
    pair_stats: Dict[str, Any],
    lang: str = "unknown",
    mention_sentence: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Reconstructs the full list of pair predictions (Gold vs Pred).
    Embeds 'vote_counts' from resampling stats directly into the rows.
    """
    gold_map = {(src, tgt): lbl for src, lbl, tgt in gold_triples}
    pred_map = {(src, tgt): lbl for src, lbl, tgt in pred_triples}
    mention_sentence = mention_sentence or {}

    # Union of all relevant pairs (Gold + Pred)
    all_pairs = set(gold_map.keys()) | set(pred_map.keys())

    pairwise_rows = []
    y_true_doc = []
    y_pred_doc = []
    sent_rel_flags = []  # "intra" | "inter" | "unknown" per pair

    for src, tgt in all_pairs:
        gold_lbl = gold_map.get((src, tgt), NOREL)
        pred_lbl = pred_map.get((src, tgt), NOREL)

        y_true_doc.append(gold_lbl)
        y_pred_doc.append(pred_lbl)

        pair_key = f"{src},{tgt}"

        # Retrieve stats if available (votes, counts)
        stats = pair_stats.get(pair_key, {})

        # Intra/inter sentence relation. "unknown" when sentence boundaries
        # aren't resolved for one or both mentions — these must NOT be folded
        # into "inter", or the inter-sentence metric silently absorbs them.
        src_sent = mention_sentence.get(src)
        tgt_sent = mention_sentence.get(tgt)
        if src_sent is None or tgt_sent is None:
            sent_rel = "unknown"
        elif src_sent == tgt_sent:
            sent_rel = "intra"
        else:
            sent_rel = "inter"
        sent_rel_flags.append(sent_rel)

        pairwise_rows.append({
            "doc_idx": doc_idx,
            "id": doc_id,
            "lang": lang,
            "pair": pair_key,
            "gold": gold_lbl,
            "pred": pred_lbl,
            "vote_counts": stats.get("vote_counts", None),
            "sentence_relation": sent_rel,
        })

    return {
        "rows": pairwise_rows,
        "y_true": y_true_doc,
        "y_pred": y_pred_doc,
        "sent_rel_flags": sent_rel_flags,
    }


def _split_intra_inter(y_true, y_pred, sent_rel_flags):
    """
    Splits parallel y_true/y_pred lists into intra-sentence and inter-sentence
    subsets. Pairs flagged "unknown" (sentence boundary couldn't be resolved
    for one or both mentions) are excluded from both — not folded into either.
    """
    intra_true, intra_pred, inter_true, inter_pred = [], [], [], []
    unknown_count = 0
    for gt, pr, rel in zip(y_true, y_pred, sent_rel_flags):
        if rel == "intra":
            intra_true.append(gt)
            intra_pred.append(pr)
        elif rel == "inter":
            inter_true.append(gt)
            inter_pred.append(pr)
        else:
            unknown_count += 1
    return intra_true, intra_pred, inter_true, inter_pred, unknown_count


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
    sent_rel_flags_all = []

    per_doc_metrics_log = []
    per_pair_predictions_log = []
    per_lang_stats = defaultdict(lambda: {"y_true": [], "y_pred": [], "sent_rel_flags": []})

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
            lang=res["lang"],
            mention_sentence=res.get("mention_sentence", {}),
        )

        y_true = alignment["y_true"]
        y_pred = alignment["y_pred"]
        sent_rel_flags = alignment["sent_rel_flags"]

        # Accumulate Global
        labels_all_true.extend(y_true)
        labels_all_pred.extend(y_pred)
        sent_rel_flags_all.extend(sent_rel_flags)

        # Accumulate Detailed Logs
        per_pair_predictions_log.extend(alignment["rows"])

        # Accumulate Per-Lang
        per_lang_stats[res["lang"]]["y_true"].extend(y_true)
        per_lang_stats[res["lang"]]["y_pred"].extend(y_pred)
        per_lang_stats[res["lang"]]["sent_rel_flags"].extend(sent_rel_flags)

        # B. Per-Doc Metrics
        mc_d = compute_multiclass_metrics(y_true, y_pred, valid_labels)
        binm_d = compute_binary_metrics(y_true, y_pred)
        intra_true_d, intra_pred_d, inter_true_d, inter_pred_d, unknown_d = _split_intra_inter(y_true, y_pred, sent_rel_flags)

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
            "binary_intra": compute_binary_metrics(intra_true_d, intra_pred_d),
            "binary_inter": compute_binary_metrics(inter_true_d, inter_pred_d),
            "sentence_unknown_pairs": unknown_d,
        })

    # --- 2. Compute Global Metrics ---
    # Note: Only compute if we have results, otherwise 0.0 defaults within metrics utils
    global_mc = compute_multiclass_metrics(labels_all_true, labels_all_pred, valid_labels)
    global_bin = compute_binary_metrics(labels_all_true, labels_all_pred)
    intra_true_g, intra_pred_g, inter_true_g, inter_pred_g, unknown_g = _split_intra_inter(labels_all_true, labels_all_pred, sent_rel_flags_all)
    global_bin_intra = compute_binary_metrics(intra_true_g, intra_pred_g)
    global_bin_inter = compute_binary_metrics(inter_true_g, inter_pred_g)
    total_pairs_g = len(labels_all_true)
    sentence_unknown_pct = (unknown_g / total_pairs_g) if total_pairs_g else 0.0

    # --- 3. Compute Per-Lang Metrics ---
    per_lang_metrics_log = {}
    for lg, data in per_lang_stats.items():
        if not data["y_true"]: continue
        intra_true_l, intra_pred_l, inter_true_l, inter_pred_l, unknown_l = _split_intra_inter(
            data["y_true"], data["y_pred"], data["sent_rel_flags"]
        )
        per_lang_metrics_log[lg] = {
            "multiclass": compute_multiclass_metrics(data["y_true"], data["y_pred"], valid_labels),
            "binary": compute_binary_metrics(data["y_true"], data["y_pred"]),
            "binary_intra": compute_binary_metrics(intra_true_l, intra_pred_l),
            "binary_inter": compute_binary_metrics(inter_true_l, inter_pred_l),
            "sentence_unknown_pairs": unknown_l,
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
        "binary_intra": global_bin_intra,
        "binary_inter": global_bin_inter,
        "sentence_unknown_pairs": unknown_g,
        "sentence_unknown_pct": sentence_unknown_pct,

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