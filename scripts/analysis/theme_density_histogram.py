#!/usr/bin/env python3
"""
Histogram of document counts by causal-relation density for Event StoryLine.

Density per document = (number of causal relations) / (number of event mentions).
One histogram per theme (ESL topic_id), each theme rendered in its own color,
sharing a common x-axis (density) range across all themes for comparability.
Loads both the train and dev splits of Nofing/EventStoryLine-0.9-standard-eci.

Usage:
    uv run scripts/analysis/theme_density_histogram.py
    uv run scripts/analysis/theme_density_histogram.py --bins 40
    uv run scripts/analysis/theme_density_histogram.py --out-dir figs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPO_ID = "Nofing/EventStoryLine-0.9-standard-eci"


def load_doc_densities() -> pd.DataFrame:
    from datasets import load_dataset

    rows = []
    for split in ("train", "dev"):
        ds = load_dataset(REPO_ID, split=split)
        for row in ds:
            num_mentions = len(row["mentions"])
            if num_mentions == 0:
                continue
            num_relations = len(row["relations"]["causes"])
            rows.append({
                "doc_id": row["id"],
                "theme": row["topic_id"],
                "split": split,
                "num_mentions": num_mentions,
                "num_relations": num_relations,
                "density": num_relations / num_mentions,
            })
    return pd.DataFrame(rows)


def plot_per_theme_histograms(df: pd.DataFrame, out_dir: Path, n_bins: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    themes = sorted(df["theme"].unique())
    cmap = plt.get_cmap("tab20", len(themes))
    colors = {theme: cmap(i) for i, theme in enumerate(themes)}

    max_density = df["density"].max()
    bin_edges = np.linspace(0, max_density, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for theme in themes:
        sub = df[df["theme"] == theme]
        counts, _ = np.histogram(sub["density"], bins=bin_edges)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(centers, counts, width=bin_width, color=colors[theme])
        ax.set_xlim(0, max_density)
        ax.set_xlabel("density (relations / mentions) per document")
        ax.set_ylabel("number of documents")
        ax.set_title(f"EventStoryLine theme {theme} (n={len(sub)} docs)")
        fig.tight_layout()
        out_path = out_dir / f"esl_theme_{theme}_density_histogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=30, help="Number of density bins (default: 30)")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "scripts" / "analysis" / "out" / "theme_density"),
        help="Directory to save the figure",
    )
    args = parser.parse_args()

    df = load_doc_densities()
    if df.empty:
        sys.exit("No documents with mentions found.")

    print(f"{len(df)} documents across {df['theme'].nunique()} themes (train+dev).")
    print(df.groupby("theme")["density"].describe().to_string())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_per_theme_histograms(df, out_dir, args.bins)


if __name__ == "__main__":
    main()
