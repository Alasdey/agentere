#!/usr/bin/env python3
"""
Produce per-pair breakdowns (correct / wrong) with document text, aggregating
predictions across multiple runs. Joins against the source HuggingFace dataset
to resolve mention text.

Usage:
    # Single run
    uv run python scripts/analysis/pair_errors.py logs/allatonce/run_XYZ.json

    # Aggregate across all runs in a directory
    uv run python scripts/analysis/pair_errors.py logs/allatonce/run_*.json

    # Latest run only
    uv run python scripts/analysis/pair_errors.py --latest logs/allatonce/

    # Write to file
    uv run python scripts/analysis/pair_errors.py --out results.json logs/allatonce/run_*.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datasets import load_dataset

NOREL_VARIANTS = frozenset({"norel", "none", "no_rel", "unknown"})


def is_causal(label: str) -> bool:
    return label.lower() not in NOREL_VARIANTS


def binary_metrics(entries: list[dict]) -> dict:
    tp = fp = fn = 0
    for e in entries:
        gold_pos = is_causal(e["gold"])
        pred_pos = is_causal(max(e["pred"], key=e["pred"].__getitem__))
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"binary_precision": precision, "binary_recall": recall, "binary_f1": f1}


def build_mentions_map(row: dict) -> dict[str, str]:
    tokens = row["tokens"]
    spans = row["spans"]
    mentions = row["mentions"]
    result = {}
    for j, mention_id in enumerate(mentions):
        parts = [tokens[idx] for idx in spans[j] if 0 <= idx < len(tokens)]
        if parts:
            result[str(mention_id)] = " ".join(parts)
    return result


def load_doc_info(repo_id: str, split: str, text_field: str) -> dict[str, dict]:
    """Returns {doc_id: {"text": ..., "mentions_map": {...}}}."""
    ds = load_dataset(repo_id, split=split)
    docs = {}
    for i in range(len(ds)):
        row = ds[i]
        doc_id = str(row["id"])
        docs[doc_id] = {
            "text": row[text_field],
            "mentions_map": build_mentions_map(row),
        }
    return docs


def get_ds_key(log_path: str) -> tuple[str, str, str]:
    """Extract (repo_id, split, text_field) from a log file."""
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["config"]
    active_ds = cfg["active_dataset"]
    ds_cfg = cfg["datasets"][active_ds]
    return (ds_cfg["repo_id"], ds_cfg["split"], ds_cfg["text_field"])


def analyze(log_paths: list[str], out_path: str | None = None, min_preds: int = 1):
    print(f"Loading {len(log_paths)} log file(s)...")

    # Deduplicate datasets; always load full split
    seen_datasets: set[tuple] = set()
    doc_info: dict[str, dict] = {}
    for p in log_paths:
        key = get_ds_key(p)
        if key not in seen_datasets:
            seen_datasets.add(key)
            repo_id, split, text_field = key
            print(f"  Fetching texts from {repo_id} ({split})...")
            doc_info.update(load_doc_info(repo_id, split, text_field))

    # key: (doc_id, raw_pair_str) → aggregated entry
    agg: dict[tuple, dict] = {}
    unresolvable: set[tuple] = set()

    for log_path in log_paths:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)

        for row in data["results"]["per_pair_predictions"]:
            doc_id = str(row["id"])
            pair_str = row["pair"]
            key = (doc_id, pair_str)

            if key in unresolvable:
                continue

            if key not in agg:
                info = doc_info[doc_id]
                mentions_map = info["mentions_map"]
                src_id, tgt_id = [p.strip() for p in pair_str.split(",", 1)]
                if src_id not in mentions_map or tgt_id not in mentions_map:
                    unresolvable.add(key)
                    continue
                agg[key] = {
                    "doc_id": doc_id,
                    "pair": f"{src_id} {mentions_map[src_id]}, {tgt_id} {mentions_map[tgt_id]}",
                    "text": info["text"],
                    "gold": row["gold"],
                    "pred": defaultdict(int),
                }

            agg[key]["pred"][row["pred"]] += 1

    if unresolvable:
        print(f"  WARNING: {len(unresolvable)} pairs skipped — mention IDs not in dataset (model hallucinations)")

    correct = []
    wrong = []

    for entry in agg.values():
        pred_counts = dict(entry["pred"])
        if sum(pred_counts.values()) < min_preds:
            continue
        majority = max(pred_counts, key=pred_counts.__getitem__)
        out_entry = {**entry, "pred": pred_counts}
        if majority == entry["gold"]:
            correct.append(out_entry)
        else:
            wrong.append(out_entry)

    all_entries = correct + wrong
    result = {
        **binary_metrics(all_entries),
        "logs": [os.path.basename(p) for p in log_paths],
        "total": len(all_entries),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "correct": correct,
        "wrong": wrong,
    }

    print(f"  Total pairs: {len(agg)}  correct: {len(correct)}  wrong: {len(wrong)}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Written to: {out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def get_active_dataset(log_path: str) -> str:
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)["config"]["active_dataset"]


def main():
    parser = argparse.ArgumentParser(description="Pair-level correct/wrong breakdown with document text")
    parser.add_argument("logs", nargs="*", help="Log JSON file(s) or directory")
    parser.add_argument("--latest", action="store_true", help="Use only the most recent log in the given dir/glob")
    parser.add_argument("--dataset", help="Filter logs to this active_dataset key (e.g. event_story_line)")
    parser.add_argument("--min-preds", type=int, default=1, help="Only include pairs with at least this many total predictions")
    parser.add_argument("--out", help="Write output JSON to this path")
    args = parser.parse_args()

    paths = []
    for p in args.logs:
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, "run_*.json")))
        elif "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)

    if not paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(set(paths), key=os.path.getmtime)

    if args.dataset:
        paths = [p for p in paths if get_active_dataset(p) == args.dataset]
        if not paths:
            print(f"No logs found for dataset '{args.dataset}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Filtered to {len(paths)} log(s) for dataset '{args.dataset}'.")

    if args.latest:
        paths = [paths[-1]]

    try:
        analyze(paths, out_path=args.out, min_preds=args.min_preds)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
