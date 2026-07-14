#!/usr/bin/env python3
"""
Binary intra-sentence metrics, recomputed two ways over mention word distance:

  1. Cumulative exclusion: drop all pairs with word_dist <= N, for N = 1..10.
  2. Exact buckets: keep only pairs with word_dist == N, for N = 1..10
     (each bucket is disjoint — the N=3 bucket does not include N=1 or N=2).

Reuses distance_histogram.extract_run() to get per-pair word_dist + confusion
(TP/FP/FN/TN) for intra-sentence pairs, then re-derives precision/recall/F1
from the confusion counts — same binary collapse as
utils.metrics.compute_binary_metrics (causes/causedby -> POS, norel -> NEG).

Usage:
    uv run scripts/analysis/intra_exclude_close_pairs.py                # latest run
    uv run scripts/analysis/intra_exclude_close_pairs.py --run <path>
    uv run scripts/analysis/intra_exclude_close_pairs.py --dataset maven_ere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.distance_histogram import LOGS_DIR, extract_run  # noqa: E402


def metrics_from_rows(df: pd.DataFrame) -> dict:
    tp = int((df["confusion"] == "TP").sum())
    fp = int((df["confusion"] == "FP").sum())
    fn = int((df["confusion"] == "FN").sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_pairs": len(df),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_pos": tp + fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Path to a specific run_*.json (default: latest in logs/allatonce)")
    parser.add_argument("--logs", default=str(LOGS_DIR), help="Log directory (default: logs/allatonce)")
    parser.add_argument("--dataset", help="Restrict to one dataset key (e.g. meci, maven_ere)")
    parser.add_argument("--thresholds", type=int, nargs="*", default=list(range(1, 11)),
                         help="Word-distance thresholds to exclude at-or-below (default: 1..10)")
    args = parser.parse_args()

    if args.run:
        run_path = Path(args.run)
    else:
        log_files = sorted(Path(args.logs).glob("run_*.json"))
        if not log_files:
            sys.exit(f"No run_*.json files found in {args.logs}")
        run_path = log_files[-1]

    print(f"Run: {run_path.name}\n")

    rows = extract_run(run_path)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No classifiable pairs in this run (dataset load failed or no per_pair_predictions).")

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    df = df[df["kind"] == "intra"]
    if df.empty:
        sys.exit("No intra-sentence pairs after filtering.")

    baseline = metrics_from_rows(df)
    print(f"{'condition':<28}{'pairs':>7}{'excluded':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'support_pos':>13}")
    print("-" * 87)
    print(f"{'baseline (all intra)':<28}{baseline['n_pairs']:>7}{0:>10}"
          f"{baseline['precision']:>11.4f}{baseline['recall']:>9.4f}{baseline['f1']:>9.4f}{baseline['support_pos']:>13}")

    for t in args.thresholds:
        sub = df[df["word_dist"] > t]
        excluded = len(df) - len(sub)
        m = metrics_from_rows(sub)
        print(f"{f'exclude word_dist <= {t}':<28}{m['n_pairs']:>7}{excluded:>10}"
              f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}{m['support_pos']:>13}")

    print(f"\n{'condition':<28}{'pairs':>7}{'':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'support_pos':>13}")
    print("-" * 87)
    for t in args.thresholds:
        sub = df[df["word_dist"] == t]
        m = metrics_from_rows(sub)
        print(f"{f'only word_dist == {t}':<28}{m['n_pairs']:>7}{'':>10}"
              f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}{m['support_pos']:>13}")


if __name__ == "__main__":
    main()
