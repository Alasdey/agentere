#!/usr/bin/env python3
"""
Ablation analysis: for a given parameter, find all run groups where every other
parameter is identical and report min/max/avg of binary_f1 per parameter value.

Usage:
    python scripts/param_ablation.py few_shot.cot_generation.enabled
    python scripts/param_ablation.py few_shot.pool_size --n 20
    python scripts/param_ablation.py model.default_model_id
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import pandas as pd


EXPERIMENT_NAME = "agentere"
METRIC = "binary_f1"

# Params to always exclude from the "context key" (vary per run by design).
IGNORE_PARAMS = {
    "experiment.tracing_name",
}

# Params to hide from the config label (too noisy or not config-identifying).
LABEL_HIDE = {
    "experiment.tracing", "experiment.tools", "encoder.path",
    "encoder.filter_norel.enabled", "encoder.filter_norel.delta", "data.labels",
}

LABEL_PRIORITY = [
    "active_dataset", "dataset.prompt", "model.default_model_id",
    "few_shot.enabled", "few_shot.n_examples", "few_shot.pool_size",
    "few_shot.selection", "experiment.resampling.enabled", "experiment.resampling.n_runs", 'few_shot.systematic', "data.binary_undirected"
]


def load_runs(tracking_uri: str, experiment_name: str, n: int) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment named '{experiment_name}' found at {tracking_uri}")
        sys.exit(1)
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=n,
        order_by=["start_time DESC"],
    )


def group_key(row: pd.Series, param_cols: list[str], ablation_col: str) -> tuple:
    return tuple(
        (col, row[col])
        for col in param_cols
        if col != ablation_col and col not in {f"params.{p}" for p in IGNORE_PARAMS}
    )


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "max": None, "avg": None}
    return {"n": len(values), "min": min(values), "max": max(values), "avg": sum(values) / len(values)}


def fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "  N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("param", help="Dotted param path, e.g. few_shot.cot_generation.enabled")
    parser.add_argument("--n", type=int, default=50, help="Max recent runs to consider (default 50)")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--min-group-size", type=int, default=2,
                        help="Min distinct ablation values in a group to be reported (default 2)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    tracking_uri = (
        str(project_root / args.tracking_uri)
        if not args.tracking_uri.startswith(("sqlite", "http", "/"))
        else args.tracking_uri
    )

    print(f"Loading last {args.n} finished runs from {tracking_uri} / {args.experiment} ...")
    df = load_runs(tracking_uri, args.experiment, args.n)
    if df.empty:
        print("No finished runs found.")
        sys.exit(0)
    print(f"  {len(df)} runs loaded.\n")

    ablation_col = f"params.{args.param}"
    if ablation_col not in df.columns:
        available = sorted(c[len("params."):] for c in df.columns if c.startswith("params."))
        print(f"Parameter '{args.param}' not found. Available params:\n  " + "\n  ".join(available))
        sys.exit(1)

    metric_col = f"metrics.{METRIC}"
    if metric_col not in df.columns:
        print(f"Metric '{METRIC}' not found in runs.")
        sys.exit(1)

    param_cols = [c for c in df.columns if c.startswith("params.")]

    # Group runs by their context key
    groups: dict[tuple, list[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        key = group_key(row, param_cols, ablation_col)
        groups[key].append(row)

    valid_groups = {
        key: rows
        for key, rows in groups.items()
        if len({r[ablation_col] for r in rows}) >= args.min_group_size
    }

    if not valid_groups:
        print(f"No groups found where '{args.param}' varies while everything else is equal.")
        print(f"(Searched {len(groups)} distinct configurations across {len(df)} runs.)")
        sys.exit(0)

    print(f"Found {len(valid_groups)} configuration(s) where '{args.param}' varies:\n")

    for group_idx, (key, rows) in enumerate(valid_groups.items(), 1):
        key_dict = {k[len("params."):]: v for k, v in dict(key).items()}
        label_parts = [
            f"{lk}={key_dict[lk]}"
            for lk in LABEL_PRIORITY
            if key_dict.get(lk) not in (None, "", "None")
        ]
        config_label = "  ".join(label_parts) or f"Config #{group_idx}"

        print(f"{'='*80}")
        print(f"Config {group_idx}: {config_label}")
        print(f"{'='*80}")
        print(f"  Runs in group: {len(rows)}")
        print()

        by_value: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            val = str(row[ablation_col])
            v = row[metric_col]
            if pd.notna(v):
                by_value[val].append(float(v))

        param_values = sorted(by_value.keys())
        val_w = max(len(v) for v in param_values + [args.param]) + 2
        col_w = 10

        header = f"  {args.param:<{val_w}}  {'n':>3}  {'min':>{col_w}}  {'max':>{col_w}}  {'avg':>{col_w}}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for val in param_values:
            s = summarise(by_value[val])
            print(f"  {val:<{val_w}}  {s['n']:>3}  {fmt(s['min']):>{col_w}}  {fmt(s['max']):>{col_w}}  {fmt(s['avg']):>{col_w}}")
        print()


if __name__ == "__main__":
    main()
