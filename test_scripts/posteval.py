#!/usr/bin/env python3
import json
import argparse
import glob
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple

def normalize_label(label: str) -> str:
    """Normalize label to Binary: 'REL' or 'NoRel'."""
    if not label or label.lower() in ("norel", "none", "unknown", "negative"):
        return "NoRel"
    return "REL"

def to_undirected_key(src: str, tgt: str) -> Tuple[str, str]:
    """Sorts IDs to ignore directionality."""
    return tuple(sorted((src, tgt)))

def compute_undirected_binary_metrics(log_file: str):
    print(f"Processing: {os.path.basename(log_file)}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract per-pair predictions
    # We use a set of (doc_id, {undirected_pair}) to track unique items
    # Value is: { 'gold_is_rel': bool, 'pred_is_rel': bool }
    
    predictions = data.get("results", {}).get("per_pair_predictions", [])
    if not predictions:
        print("  [WARN] No per-pair predictions found in log.")
        return

    # Map: (doc_id, undirected_pair_tuple) -> state
    pair_map: Dict[Tuple[str, Tuple[str, str]], Dict[str, bool]] = defaultdict(lambda: {"gold": False, "pred": False})

    for row in predictions:
        doc_id = str(row.get("id"))
        pair_str = row.get("pair", "")
        if "," not in pair_str:
            continue
            
        src, tgt = [x.strip() for x in pair_str.split(",", 1)]
        
        # Current Row Labels
        row_gold = normalize_label(row.get("gold"))
        row_pred = normalize_label(row.get("pred"))

        # Undirected Key
        u_key = to_undirected_key(src, tgt)
        full_key = (doc_id, u_key)

        # Logic: 
        # "Prediction of causality in one way only is sufficient to predict causality both ways"
        # Since we are flattening to undirected, if we see (A,B)=REL OR (B,A)=REL, the undirected edge {A,B} is REL.
        if row_gold == "REL":
            pair_map[full_key]["gold"] = True
        
        if row_pred == "REL":
            pair_map[full_key]["pred"] = True

    # 2. Compute Metrics
    tp = 0
    fp = 0
    fn = 0

    for key, state in pair_map.items():
        g = state["gold"]
        p = state["pred"]

        if g and p:
            tp += 1
        elif not g and p:
            fp += 1
        elif g and not p:
            fn += 1
        # True Negatives (not g and not p) are ignored in F1

    # 3. Formula
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Total Undirected Pairs Processed: {len(pair_map)}")
    print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  Binary Undirected Precision: {precision:.4f}")
    print(f"  Binary Undirected Recall:    {recall:.4f}")
    print(f"  Binary Undirected F1:        {f1:.4f}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Calculate Undirected Binary Metrics from Logs")
    parser.add_argument("--logs", type=str, default="logs", help="Directory containing run_*.json files")
    parser.add_argument("--latest", action="store_true", help="Only process the most recent log file")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.logs, "run_*.json"))
    if not files:
        print(f"No JSON logs found in {args.logs}")
        return

    # Sort by time
    files.sort(key=os.path.getmtime)

    if args.latest:
        files = [files[-1]]

    for f in files:
        try:
            compute_undirected_binary_metrics(f)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()