"""
recompute_metrics.py

Recomputes directed metrics in MECI log files from per_pair_predictions,
skipping gold="unannotated" pairs. NoRel is still excluded from the
global macro/micro metrics (same convention as the original run).

The script rewrites: per_label, macro_f1, micro_*, binary, total_pairs,
per_doc_metrics, and per_lang_metrics inside results{}.
per_pair_predictions itself is left unchanged.

Usage:
    python recompute_metrics.py                        # all MECI logs in ../logs/
    python recompute_metrics.py path/to/run.json ...  # specific files
    python recompute_metrics.py --dry-run              # print diff, no writes
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import compute_multiclass_metrics, compute_binary_metrics

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def recompute_from_perpair(
    per_pair: list[dict[str, Any]],
    valid_labels: list[str],
) -> dict[str, Any]:
    """
    Recomputes all metrics from per_pair_predictions, ignoring unannotated pairs.

    Returns a dict with the same keys that reporting.generate_run_report() produces
    (minus per_pair_predictions and resampling, which are left untouched).
    """
    global_true: list[str] = []
    global_pred: list[str] = []
    per_lang: dict[str, dict] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    per_doc: dict[tuple, dict] = {}  # (doc_idx, id) -> {y_true, y_pred}

    for entry in per_pair:
        if entry.get("gold") == "unannotated":
            continue

        gold = entry["gold"]
        pred = entry["pred"]
        lang = entry.get("lang", "unknown")
        doc_key = (entry["doc_idx"], entry["id"])

        global_true.append(gold)
        global_pred.append(pred)
        per_lang[lang]["y_true"].append(gold)
        per_lang[lang]["y_pred"].append(pred)

        if doc_key not in per_doc:
            per_doc[doc_key] = {"doc_idx": entry["doc_idx"], "id": entry["id"], "y_true": [], "y_pred": []}
        per_doc[doc_key]["y_true"].append(gold)
        per_doc[doc_key]["y_pred"].append(pred)

    # Global
    global_mc = compute_multiclass_metrics(global_true, global_pred, valid_labels)
    global_bin = compute_binary_metrics(global_true, global_pred)

    # Per-doc
    per_doc_metrics = []
    for doc_key in sorted(per_doc, key=lambda k: k[0]):
        dd = per_doc[doc_key]
        mc = compute_multiclass_metrics(dd["y_true"], dd["y_pred"], valid_labels)
        bm = compute_binary_metrics(dd["y_true"], dd["y_pred"])
        per_doc_metrics.append({
            "doc_idx": dd["doc_idx"],
            "id": dd["id"],
            "pairs": mc["total"],
            "macro_f1": mc["macro_f1"],
            "micro_f1": mc["micro_f1"],
            "micro_precision": mc["micro_precision"],
            "micro_recall": mc["micro_recall"],
            "per_label": mc["per_label"],
            "binary": bm,
        })

    # Per-lang
    per_lang_metrics = {}
    for lg, data in per_lang.items():
        mc = compute_multiclass_metrics(data["y_true"], data["y_pred"], valid_labels)
        bm = compute_binary_metrics(data["y_true"], data["y_pred"])
        per_lang_metrics[lg] = {
            "multiclass": mc,
            "binary": bm,
            "total_pairs": mc["total"],
        }

    return {
        "per_label": global_mc["per_label"],
        "macro_f1": global_mc["macro_f1"],
        "micro_precision": global_mc["micro_precision"],
        "micro_recall": global_mc["micro_recall"],
        "micro_f1": global_mc["micro_f1"],
        "total_pairs": global_mc["total"],
        "binary": global_bin,
        "per_doc_metrics": per_doc_metrics,
        "per_lang_metrics": per_lang_metrics,
    }


def process_log(path: str, dry_run: bool) -> dict[str, Any] | None:
    """
    Loads a log file, recomputes metrics, and writes back unless dry_run.
    Returns a summary dict for display, or None if the file is not applicable.
    """
    with open(path) as f:
        data = json.load(f)

    ppp = data.get("results", {}).get("per_pair_predictions")
    if not isinstance(ppp, list):
        return None

    cfg = data.get("config", {})
    # Prefer labels from config; fall back to whatever keys appear in the original
    # per_label (which reflects exactly what valid_labels were passed at run time).
    valid_labels = (
        cfg.get("datasets", {}).get("meci", {}).get("labels")
        or list(data["results"]["per_label"].keys())
    )

    old_r = data["results"]
    new_metrics = recompute_from_perpair(ppp, valid_labels)

    # How many unannotated pairs are in this file
    unannotated = sum(1 for e in ppp if e.get("gold") == "unannotated")

    summary = {
        "path": path,
        "unannotated_excluded": unannotated,
        "old": {
            "total_pairs": old_r.get("total_pairs"),
            "macro_f1": old_r.get("macro_f1"),
            "micro_f1": old_r.get("micro_f1"),
            "binary_f1": old_r.get("binary", {}).get("f1"),
        },
        "new": {
            "total_pairs": new_metrics["total_pairs"],
            "macro_f1": new_metrics["macro_f1"],
            "micro_f1": new_metrics["micro_f1"],
            "binary_f1": new_metrics["binary"]["f1"],
        },
    }

    if not dry_run:
        # Update results in-place, keep per_pair_predictions and resampling
        for key, val in new_metrics.items():
            old_r[key] = val
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return summary


def find_meci_logs(logs_dir: str) -> list[str]:
    paths = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            if cfg.get("active_dataset") != "meci":
                continue
            ppp = data.get("results", {}).get("per_pair_predictions")
            if isinstance(ppp, list):
                paths.append(path)
        except Exception:
            continue
    return paths


def fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Log JSON files to process (default: all MECI logs)")
    parser.add_argument("--dry-run", action="store_true", help="Print diffs without modifying files")
    args = parser.parse_args()

    files = args.files if args.files else find_meci_logs(LOGS_DIR)
    if not files:
        print("No MECI log files found.", file=sys.stderr)
        sys.exit(1)

    prefix = "[dry]" if args.dry_run else "[updated]"
    header = f"{'file':<55}  {'excl':>6}  {'macro_f1':>10}  {'micro_f1':>10}  {'bin_f1':>8}  {'pairs':>7}"
    print(header)
    print("-" * len(header))

    for path in files:
        s = process_log(path, dry_run=args.dry_run)
        if s is None:
            continue
        name = os.path.basename(s["path"])
        excl = s["unannotated_excluded"]
        old, new = s["old"], s["new"]

        macro_change = f"{fmt(old['macro_f1'])} -> {fmt(new['macro_f1'])}"
        micro_change = f"{fmt(old['micro_f1'])} -> {fmt(new['micro_f1'])}"
        bin_change   = f"{fmt(old['binary_f1'])} -> {fmt(new['binary_f1'])}"
        pairs_change = f"{old['total_pairs']} -> {new['total_pairs']}"

        print(f"{prefix} {name:<52}  {excl:>6}  {macro_change:>22}  {micro_change:>22}  {bin_change:>18}  {pairs_change:>15}")

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
