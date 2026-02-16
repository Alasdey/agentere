#!/usr/bin/env python3
"""
Summarize run_*.json logs into an Excel sheet.

Handles the nested config structure with active_dataset, experiment block,
encoder_predictions, etc.

Extracted columns (all optional / robust to missing keys):
  - run_id, status, timestamp_utc, file
  - model_id, temperature
  - active_dataset, dataset_name, split, max_examples
  - langsmith_project (tracing_name)
  - tools_enabled, tools
  - encoder_filter_norel_enabled, encoder_filter_norel_delta
  - micro_precision, micro_recall, micro_f1, macro_f1, total_pairs
  - binary_precision, binary_recall, binary_f1, binary_support_pos
  - per-label metrics (e.g. CAUSE_precision, CAUSE_recall, …)
  - skipped_docs

Usage:
  python summarize_runs.py --logs ./logs --out runs_summary.xlsx
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def safe_get(d: Any, path: List[str], default=None):
    """Walk *path* into nested dicts; return *default* on any miss."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def flatten_per_label(results: Dict[str, Any]) -> Dict[str, Any]:
    """Expand results.per_label into flat columns."""
    out: Dict[str, Any] = {}
    per_label = results.get("per_label") or {}
    for label, metrics in per_label.items():
        if not isinstance(metrics, dict):
            continue
        for metric in ("precision", "recall", "f1", "support"):
            out[f"{label}_{metric}"] = metrics.get(metric)
    return out


def flatten_binary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Expand results.binary into flat columns."""
    out: Dict[str, Any] = {}
    binary = results.get("binary") or {}
    if isinstance(binary, dict):
        for metric in ("precision", "recall", "f1", "support_pos"):
            out[f"binary_{metric}"] = binary.get(metric)
    return out


def get_active_dataset_cfg(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config dict for the active dataset (or {})."""
    active = safe_get(data, ["config", "active_dataset"])
    if not active:
        return {}
    return safe_get(data, ["config", "datasets", active]) or {}


# ---------------------------------------------------------------------------
# row extraction
# ---------------------------------------------------------------------------

def extract_row(data: Dict[str, Any], fname: str) -> Dict[str, Any]:
    ds_cfg = get_active_dataset_cfg(data)
    experiment = safe_get(data, ["config", "experiment"]) or {}
    results = data.get("results") or {}

    # Tools: only list them when enable_tools is truthy
    tools_enabled = experiment.get("enable_tools", False)
    tools_list = experiment.get("tools") or []
    tools_str = ", ".join(tools_list) if tools_enabled else ""

    row: Dict[str, Any] = {
        # identifiers
        "file":               os.path.basename(fname),
        "run_id":             data.get("run_id"),
        "status":             data.get("status"),
        "timestamp_utc":      data.get("timestamp_utc"),

        # model
        "model_id":           safe_get(data, ["config", "model", "default_model_id"]),
        "temperature":        safe_get(data, ["config", "model", "temperature"]),

        # active dataset
        "active_dataset":     safe_get(data, ["config", "active_dataset"]),
        "dataset_name":       ds_cfg.get("name"),
        "split":              ds_cfg.get("split"),
        "max_examples":       ds_cfg.get("max_examples"),

        # experiment / tracing
        "langsmith_project":  experiment.get("tracing_name"),
        "tools_enabled":      tools_enabled,
        "tools":              tools_str,

        # encoder predictions – filter_norel
        "encoder_filter_norel_enabled": safe_get(data, ["config", "encoder_predictions", "filter_norel", "enabled"]),
        "encoder_filter_norel_delta":   safe_get(data, ["config", "encoder_predictions", "filter_norel", "delta"]),

        # aggregate metrics
        "micro_precision":    results.get("micro_precision"),
        "micro_recall":       results.get("micro_recall"),
        "micro_f1":           results.get("micro_f1"),
        "macro_f1":           results.get("macro_f1"),
        "total_pairs":        results.get("total_pairs"),

        # skipped docs (check two possible keys)
        "skipped_docs":       results.get("skipped_docs") or results.get("skipped") or 0,
    }

    # binary metrics
    row.update(flatten_binary(results))

    # per-label metrics
    row.update(flatten_per_label(results))

    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs",
                    help="Folder containing run_*.json files")
    ap.add_argument("--out", default="runs_summary.xlsx",
                    help="Path to output Excel file")
    args = ap.parse_args()

    pattern = os.path.join(args.logs, "run_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(extract_row(data, fp))
        except Exception as e:
            rows.append({
                "file": os.path.basename(fp),
                "parse_error": str(e),
            })

    df = pd.DataFrame(rows)

    # ---- column ordering: important cols first, then per-label cols ----
    preferred = [
        "file", "run_id", "status", "timestamp_utc",
        "model_id", "temperature",
        "active_dataset", "dataset_name", "split", "max_examples",
        "langsmith_project", "tools_enabled", "tools",
        "encoder_filter_norel_enabled", "encoder_filter_norel_delta",
        "micro_precision", "micro_recall", "micro_f1",
        "macro_f1", "total_pairs",
        "binary_precision", "binary_recall", "binary_f1", "binary_support_pos",
        "skipped_docs",
    ]

    cols_front = [c for c in preferred if c in df.columns]
    cols_rest  = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + cols_rest]

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="runs")

    print(f"Wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()