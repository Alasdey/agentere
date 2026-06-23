#!/usr/bin/env python3
"""
Histogram of TP / FP / FN relation predictions by mention distance
(in words and in sentences), split by intra- vs. inter-sentence pairs.
True negatives (gold norel, pred norel) are dropped.

The binary collapse treats causes and causedby both as the positive class
(same as utils.metrics.compute_binary_metrics), vs. norel as negative.

Usage:
    uv run scripts/analysis/distance_histogram.py
    uv run scripts/analysis/distance_histogram.py --since-days 7
    uv run scripts/analysis/distance_histogram.py --dataset meci --out-dir figs/
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import _NOREL_NORM  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs" / "allatonce"
RUN_TS_RE = re.compile(r"run_(\d{4}-\d{2}-\d{2})T")


def binarize(label: str) -> str:
    return "NEG" if str(label).lower() in _NOREL_NORM else "POS"


# ---------------------------------------------------------------------------
# Dataset loading and indexing: mention_id -> (sent_idx, word_pos)
# ---------------------------------------------------------------------------

_DATASET_CACHE: Dict[Tuple[str, str], Dict[str, Dict[str, Tuple[int, int]]]] = {}


def mention_sentence(tok: int, sentences: List[List[int]]) -> int:
    for idx, (s, e) in enumerate(sentences):
        if s <= tok < e:
            return idx
    return len(sentences)


def load_doc_index(repo_id: str, split: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    key = (repo_id, split)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split)
    index: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for row in ds:
        doc_id: str = row["id"]
        mentions: List[str] = row["mentions"]
        spans: List[List[int]] = row["spans"]
        sentences: List[List[int]] = row["sentences"]

        doc_map: Dict[str, Tuple[int, int]] = {}
        for m, sp in zip(mentions, spans):
            if not sp:
                continue
            tok = sp[0]
            doc_map[m] = (mention_sentence(tok, sentences), tok)
        index[doc_id] = doc_map

    _DATASET_CACHE[key] = index
    return index


def dataset_key_to_config(ds_key: str, datasets_cfg: dict) -> Optional[Tuple[str, str]]:
    ds_cfg = datasets_cfg.get(ds_key)
    if not ds_cfg:
        return None
    repo_id = ds_cfg.get("repo_id")
    split = ds_cfg.get("split", "test")
    if not repo_id:
        return None
    return repo_id, split


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------

def extract_run(log_path: Path) -> List[Dict]:
    with open(log_path) as f:
        data = json.load(f)

    config = data.get("config") or {}
    active_ds = config.get("active_dataset")
    if not active_ds:
        return []

    ds_info = dataset_key_to_config(active_ds, config.get("datasets") or {})
    if not ds_info:
        return []

    repo_id, split = ds_info
    try:
        doc_index = load_doc_index(repo_id, split)
    except Exception as exc:
        print(f"  Skipping {log_path.name}: could not load dataset — {exc}", file=sys.stderr)
        return []

    per_pair = (data.get("results") or {}).get("per_pair_predictions") or []
    rows: List[Dict] = []

    for pred in per_pair:
        if pred.get("gold") == "unannotated":
            continue
        parts = pred["pair"].split(",", 1)
        if len(parts) != 2:
            continue
        mention_to_pos = doc_index.get(pred["id"], {})
        src = mention_to_pos.get(parts[0].strip())
        tgt = mention_to_pos.get(parts[1].strip())
        if src is None or tgt is None:
            continue

        gold_pos = binarize(pred["gold"]) == "POS"
        pred_pos = binarize(pred["pred"]) == "POS"
        if gold_pos and pred_pos:
            confusion = "TP"
        elif gold_pos and not pred_pos:
            confusion = "FN"
        elif not gold_pos and pred_pos:
            confusion = "FP"
        else:
            confusion = "TN"

        sent_dist = abs(src[0] - tgt[0])
        word_dist = abs(src[1] - tgt[1])
        rows.append({
            "dataset": active_ds,
            "doc_id": pred["id"],
            "num_mentions": len(mention_to_pos),
            "kind": "intra" if sent_dist == 0 else "inter",
            "word_dist": word_dist,
            "sent_dist": sent_dist,
            "confusion": confusion,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histogram of correct/incorrect predictions by word/sentence distance, intra vs inter."
    )
    parser.add_argument("--logs", default=str(LOGS_DIR), help="Log directory (default: logs/allatonce)")
    parser.add_argument("--since-days", type=int, default=7, help="Only include runs from the last N days (default: 7)")
    parser.add_argument("--dataset", help="Restrict to one dataset key (e.g. meci, maven_ere)")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "scripts" / "analysis" / "out" / "histogram"), help="Directory to save PNG figures")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    log_files = sorted(log_dir.glob("run_*.json"))
    if not log_files:
        sys.exit(f"No run_*.json files found in {log_dir}")

    cutoff = datetime.date.today() - datetime.timedelta(days=args.since_days)
    selected = []
    for f in log_files:
        m = RUN_TS_RE.match(f.name)
        if m and datetime.date.fromisoformat(m.group(1)) >= cutoff:
            selected.append(f)

    if not selected:
        sys.exit(f"No run logs found from the last {args.since_days} days in {log_dir}")

    print(f"Using {len(selected)} run log(s) from the last {args.since_days} day(s):")
    for f in selected:
        print(f"  {f.name}")
    print()

    all_rows: List[Dict] = []
    for log_path in selected:
        all_rows.extend(extract_run(log_path))

    df = pd.DataFrame(all_rows)
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if df.empty:
        sys.exit("No classifiable pairs after filtering.")

    # True negatives (gold norel, pred norel) are dropped — they dilute the
    # error-distance picture and aren't part of the TP/FP/FN breakdown requested.
    df = df[df["confusion"] != "TN"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report(df, "all datasets", out_dir)
    report_density(df, "all datasets", out_dir)
    report_density_pairs(df, "all datasets", out_dir)
    report_num_mentions(df, "all datasets", out_dir)
    for dataset_name, sub_df in df.groupby("dataset"):
        ds_dir = out_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        report(sub_df, dataset_name, ds_dir)
        report_density(sub_df, dataset_name, ds_dir)
        report_density_pairs(sub_df, dataset_name, ds_dir)
        report_num_mentions(sub_df, dataset_name, ds_dir)


CONFUSION_COLORS = [("TP", "limegreen"), ("FP", "red"), ("FN", "dodgerblue")]


def plot_grouped_bar_histogram(
    ax, sub: pd.DataFrame, value_col: str, title: str, integer_valued: bool = True, weight_col: Optional[str] = None
) -> float:
    """Plots TP/FP/FN as side-by-side grouped bars over value_col, binned adaptively. Returns max value plotted.

    integer_valued=True (word/sentence distances) uses one bin per integer when the
    range is small, so bars align with whole-number ticks. Continuous data (e.g. a
    density ratio) should pass integer_valued=False to always get a fixed bin count,
    since "one bin per unit" is meaningless and far too coarse for fractional values.

    weight_col, if given, sums that column per bin instead of counting rows — e.g. so
    a single document row contributes its actual relation count rather than 1.
    """
    import numpy as np

    max_val = float(sub[value_col].max())
    if integer_valued:
        # Fine-grained integer-ish bins when the range is small enough to stay
        # readable, otherwise a fixed bin count so sparse long tails don't
        # produce thousands of near-empty bins.
        n_bins = int(max_val) + 1 if max_val <= 100 else 60
        bin_edges = np.linspace(0, max_val + (1 if max_val <= 100 else 0), n_bins + 1)
    else:
        n_bins = 30
        bin_edges = np.linspace(0, max_val if max_val > 0 else 1, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_groups = len(CONFUSION_COLORS)
    bar_width = bin_width / (n_groups + 1)

    for i, (confusion, color) in enumerate(CONFUSION_COLORS):
        rows = sub.loc[sub["confusion"] == confusion]
        weights = rows[weight_col] if weight_col else None
        counts, _ = np.histogram(rows[value_col], bins=bin_edges, weights=weights)
        offset = (i - (n_groups - 1) / 2) * bar_width
        ax.bar(centers + offset, counts, width=bar_width, label=confusion, color=color)
    ax.set_title(title)
    ax.set_ylabel("relation count" if weight_col else "count")
    ax.legend()
    return max_val


def per_doc_confusion_counts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, doc_id) with TP/FP/FN relation counts and num_mentions."""
    doc_counts = (
        df.groupby(["dataset", "doc_id", "confusion"], observed=True)
        .size()
        .unstack("confusion", fill_value=0)
    )
    for c in ["TP", "FP", "FN"]:
        if c not in doc_counts.columns:
            doc_counts[c] = 0
    num_mentions = df.groupby(["dataset", "doc_id"], observed=True)["num_mentions"].first()
    doc_counts = doc_counts.join(num_mentions)
    return doc_counts[doc_counts["num_mentions"] > 0]


def _report_density_generic(
    doc_counts: pd.DataFrame, denominator: pd.Series, label: str, out_dir: Path,
    file_prefix: str, verbose: bool, print_label: str, xlabel: str, fname: str,
) -> None:
    density_rows = []
    for confusion in ["TP", "FP", "FN"]:
        density = doc_counts[confusion] / denominator
        density_rows.append(pd.DataFrame({
            "density": density,
            "confusion": confusion,
            "relation_count": doc_counts[confusion],
        }))
    density_df = pd.concat(density_rows, ignore_index=True)
    density_df = density_df[density_df["density"].notna()]

    if verbose:
        print(f"\n=== {print_label} — {label} ===")
        print(density_df.groupby("confusion")["density"].describe().to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Weighted by relation_count so each document contributes its actual number of
    # TP/FP/FN relations to the bar height, not just 1 — keeps the total comparable
    # to the pair-level word/sentence distance histograms.
    plot_grouped_bar_histogram(
        ax, density_df, "density", print_label,
        integer_valued=False, weight_col="relation_count",
    )
    ax.set_xlabel(xlabel)
    fig.suptitle(f"{print_label} — {label}")
    fig.tight_layout()
    out_path = out_dir / f"{file_prefix}{fname}"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"\nSaved figure → {out_path}")


def report_density(df: pd.DataFrame, label: str, out_dir: Path, file_prefix: str = "", verbose: bool = True) -> None:
    """Density of positive relations (TP/FP/FN) per document, normalized by mention count."""
    doc_counts = per_doc_confusion_counts(df)
    _report_density_generic(
        doc_counts, doc_counts["num_mentions"], label, out_dir, file_prefix, verbose,
        print_label="Density of positive relations per document (per mention)",
        xlabel="positive relations / num. mentions in document",
        fname="density_per_document_histogram.png",
    )


def report_density_pairs(df: pd.DataFrame, label: str, out_dir: Path, file_prefix: str = "", verbose: bool = True) -> None:
    """Density of positive relations (TP/FP/FN) per document, normalized by the number
    of ordered mention pairs (n * (n-1)) — i.e. relation count over candidate-pair count."""
    doc_counts = per_doc_confusion_counts(df)
    doc_counts = doc_counts[doc_counts["num_mentions"] >= 2]
    denominator = doc_counts["num_mentions"] * (doc_counts["num_mentions"] - 1)
    _report_density_generic(
        doc_counts, denominator, label, out_dir, file_prefix, verbose,
        print_label="Density of positive relations per document (per ordered mention pair)",
        xlabel="positive relations / (num. mentions * (num. mentions - 1))",
        fname="density_per_pair_histogram.png",
    )


def report_num_mentions(df: pd.DataFrame, label: str, out_dir: Path, file_prefix: str = "", verbose: bool = True) -> None:
    """TP/FP/FN relation counts by number of mentions in the document."""
    doc_counts = per_doc_confusion_counts(df)

    rows = []
    for confusion in ["TP", "FP", "FN"]:
        rows.append(pd.DataFrame({
            "num_mentions": doc_counts["num_mentions"],
            "confusion": confusion,
            "relation_count": doc_counts[confusion],
        }))
    mentions_df = pd.concat(rows, ignore_index=True)

    if verbose:
        print(f"\n=== Relations by number of mentions in document — {label} ===")
        print(mentions_df.groupby("confusion")["num_mentions"].describe().to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_grouped_bar_histogram(
        ax, mentions_df, "num_mentions", "Relations by document mention count",
        integer_valued=True, weight_col="relation_count",
    )
    ax.set_xlabel("number of mentions in document")
    fig.suptitle(f"Relations by number of mentions in document — {label}")
    fig.tight_layout()
    out_path = out_dir / f"{file_prefix}num_mentions_per_document_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"\nSaved figure → {out_path}")


def report(df: pd.DataFrame, label: str, out_dir: Path, file_prefix: str = "", verbose: bool = True) -> None:
    if verbose:
        print(f"\n############ Dataset: {label} ############")

    # ------------------------------------------------------------------
    # Printed bucketed counts
    # ------------------------------------------------------------------
    word_bins = [0, 1, 3, 5, 10, 20, 40, 80, 160, 320, 640, float("inf")]
    word_labels = ["0", "1-2", "3-4", "5-9", "10-19", "20-39", "40-79", "80-159", "160-319", "320-639", "640+"]
    sent_bins = [0, 1, 2, 3, 5, 10, 20, 40, float("inf")]
    sent_labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20-39", "40+"]

    df = df.copy()
    df["word_bucket"] = pd.cut(df["word_dist"], bins=word_bins, labels=word_labels, right=False)
    df["sent_bucket"] = pd.cut(df["sent_dist"], bins=sent_bins, labels=sent_labels, right=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    if verbose:
        for dist_col, dist_label in [("word_bucket", "Word distance"), ("sent_bucket", "Sentence distance")]:
            print(f"\n=== {dist_label} histogram (counts) ===")
            table = (
                df.groupby(["kind", dist_col, "confusion"], observed=True)
                .size()
                .unstack("confusion", fill_value=0)
            )
            table = table[[c for c in ["TP", "FP", "FN"] if c in table.columns]]
            print(table.to_string())

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sentence distance: intra-sentence pairs all land at distance 0, so they're
    # folded into the same single panel as inter-sentence pairs rather than
    # split into their own subplot.
    plot_specs = [("word_dist", "Word distance between mentions", "word_distance_histogram.png", ["intra", "inter"]),
                  ("sent_dist", "Sentence distance between mentions", "sentence_distance_histogram.png", ["all"])]

    for dist_col, title, fname, kinds in plot_specs:
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), sharey=False, squeeze=False)
        axes = axes[0]
        for ax, kind in zip(axes, kinds):
            sub = df if kind == "all" else df[df["kind"] == kind]
            if sub.empty:
                ax.set_title(f"{kind} (no data)")
                continue
            panel_title = "all pairs (intra + inter)" if kind == "all" else f"{kind}-sentence pairs"
            max_val = plot_grouped_bar_histogram(ax, sub, dist_col, panel_title)
            ax.set_title(f"{panel_title} (max={int(max_val)})")
            ax.set_xlabel(title)
        fig.suptitle(f"{title} — {label}")
        fig.tight_layout()
        out_path = out_dir / f"{file_prefix}{fname}"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        if verbose:
            print(f"\nSaved figure → {out_path}")


def generate_run_histograms(run_json_path: Path, out_dir: Path, stem: str) -> None:
    """Generates all histogram artifacts for a single completed run, as sibling files
    next to its other logged artifacts (named '{stem}.<histogram_name>.png')."""
    try:
        rows = extract_run(run_json_path)
    except Exception as exc:
        print(f"  Histogram generation skipped: could not extract run data — {exc}", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df[df["confusion"] != "TN"]
    if df.empty:
        return

    label = df["dataset"].iloc[0]
    file_prefix = f"{stem}."
    out_dir.mkdir(parents=True, exist_ok=True)

    report(df, label, out_dir, file_prefix=file_prefix, verbose=False)
    report_density(df, label, out_dir, file_prefix=file_prefix, verbose=False)
    report_density_pairs(df, label, out_dir, file_prefix=file_prefix, verbose=False)
    report_num_mentions(df, label, out_dir, file_prefix=file_prefix, verbose=False)
    print(f"Histogram artifacts saved alongside run log (prefix: {file_prefix})")


if __name__ == "__main__":
    main()
