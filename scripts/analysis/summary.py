#!/usr/bin/env python3
"""
Summarize run_*.json logs into an Excel sheet.

Handles the nested config structure with active_dataset, experiment block,
encoder_predictions, etc.

Extracted columns (all optional / robust to missing keys):
  - run_id, status, timestamp_utc, file
  - model_id, temperature
  - active_dataset, dataset_name, split, max_examples, prompt
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


def flatten_per_lang(results: Dict[str, Any]) -> Dict[str, Any]:
    """Expand results.per_lang_metrics into flat columns."""
    out: Dict[str, Any] = {}
    per_lang = results.get("per_lang_metrics") or {}
    for lang, lm in per_lang.items():
        if not isinstance(lm, dict):
            continue
        mc = lm.get("multiclass") or {}
        out[f"{lang}_micro_f1"]        = mc.get("micro_f1")
        out[f"{lang}_macro_f1"]        = mc.get("macro_f1")
        out[f"{lang}_micro_precision"] = mc.get("micro_precision")
        out[f"{lang}_micro_recall"]    = mc.get("micro_recall")
        out[f"{lang}_total_pairs"]     = lm.get("total_pairs")
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
        "neg_fact":           ds_cfg.get("neg_fact"),
        "prompt":             ds_cfg.get("prompt"),

        # experiment / tracing
        "langsmith_project":  experiment.get("tracing_name"),
        "tools_enabled":      tools_enabled,
        "tools":              tools_str,

        # few-shot
        "few_shot_enabled":    safe_get(data, ["config", "few_shot", "enabled"]),
        "few_shot_n":          safe_get(data, ["config", "few_shot", "n_examples"]),
        "few_shot_split":      safe_get(data, ["config", "few_shot", "split"]),
        "few_shot_systematic": safe_get(data, ["config", "few_shot", "systematic"]),
        "few_shot_selection":  safe_get(data, ["config", "few_shot", "selection"]),
        "few_shot_bert_model": safe_get(data, ["config", "few_shot", "bert_model"]),

        # reprompt
        "reprompt_systematic": safe_get(data, ["config", "reprompt", "systematic"]),

        # experiment timing / reliability
        "timeout": experiment.get("timeout"),
        "retries": experiment.get("retries"),

        # resampling
        "resampling_enabled":     safe_get(data, ["config", "experiment", "resampling", "enabled"]),
        "resampling_n_runs":      safe_get(data, ["config", "experiment", "resampling", "n_runs"]),
        "resampling_tie_breaking": safe_get(data, ["config", "experiment", "resampling", "tie_breaking"]),

        # encoder predictions – filter_norel
        "encoder_filter_norel_enabled": safe_get(data, ["config", "encoder", "filter_norel", "enabled"]),
        "encoder_filter_norel_delta":   safe_get(data, ["config", "encoder", "filter_norel", "delta"]),

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

    # per-language metrics
    row.update(flatten_per_lang(results))

    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")


def parse_log(fp: str) -> Dict[str, Any]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return extract_row(data, fp)
    except Exception as e:
        return {"file": os.path.basename(fp), "parse_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(_PROJECT_ROOT, "logs"),
                    help="Root logs folder (searched recursively for run_*.json)")
    ap.add_argument("--out", default="runs_summary.xlsx",
                    help="Path to output Excel file")
    ap.add_argument("--mode", choices=["append", "update", "all"], default="append",
                    help="append: new files only (default); update: refresh existing rows only; all: both")
    args = ap.parse_args()

    # ---- load existing summary (if any) ----
    existing_df = None
    known_files: set = set()
    if os.path.exists(args.out):
        existing_df = pd.read_excel(args.out, sheet_name="runs")
        if "file" in existing_df.columns:
            known_files = set(existing_df["file"].dropna())
        print(f"Loaded existing summary: {len(existing_df)} rows, {len(known_files)} known files")

    # ---- index all log files by basename ----
    pattern = os.path.join(args.logs, "**", "run_*.json")
    all_files = sorted(glob.glob(pattern, recursive=True))
    log_index = {os.path.basename(fp): fp for fp in all_files}

    # ---- update rows for files already in the summary ----
    updated_count = 0
    if args.mode in ("update", "all") and existing_df is not None and "file" in existing_df.columns:
        refreshed: Dict[str, Dict[str, Any]] = {}
        for basename, fp in log_index.items():
            if basename in known_files:
                refreshed[basename] = parse_log(fp)
        if refreshed:
            for basename, new_row in refreshed.items():
                mask = existing_df["file"] == basename
                for col, val in new_row.items():
                    if col not in existing_df.columns:
                        existing_df[col] = None
                    existing_df.loc[mask, col] = val
            updated_count = len(refreshed)
            print(f"Updated {updated_count} existing rows from logs")

    # ---- append genuinely new files ----
    new_files = [fp for bn, fp in log_index.items() if bn not in known_files] \
        if args.mode in ("append", "all") else []
    new_rows: List[Dict[str, Any]] = [parse_log(fp) for fp in new_files]

    if not new_rows and not updated_count:
        print("No new log files found and nothing to update — summary is already up to date.")
        return

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df

    # ---- column ordering: important cols first, then per-label cols ----
    preferred = [
        "file", "run_id", "status", "timestamp_utc",
        "model_id", "temperature",
        "active_dataset", "dataset_name", "split", "max_examples", "neg_fact", "prompt",
        "langsmith_project", "tools_enabled", "tools",
        "few_shot_enabled", "few_shot_n", "few_shot_split", "few_shot_systematic",
        "few_shot_selection", "few_shot_bert_model",
        "reprompt_systematic",
        "timeout", "retries",
        "resampling_enabled", "resampling_n_runs", "resampling_tie_breaking",
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

    parts = []
    if new_rows:
        parts.append(f"appended {len(new_rows)} new rows")
    if updated_count:
        parts.append(f"refreshed {updated_count} existing rows")
    print(f"{', '.join(parts).capitalize()} → {args.out} ({len(df)} total)")


if __name__ == "__main__":
    main()