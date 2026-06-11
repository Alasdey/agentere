#!/usr/bin/env python3
"""
Compare binary metrics (precision, recall, F1) between intra- and inter-sentence
pairs across finished run logs.

All datasets are expected to have a `sentences` column in their HF schema
([[start_tok, end_tok), ...]).  MECI gained this column after being rebuilt
with the updated dataprep.

Usage:
    uv run scripts/analysis/intra_inter_compare.py
    uv run scripts/analysis/intra_inter_compare.py --dataset meci
    uv run scripts/analysis/intra_inter_compare.py --out results.csv
    uv run scripts/analysis/intra_inter_compare.py --logs path/to/logs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import compute_binary_metrics  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs" / "allatonce"


# ---------------------------------------------------------------------------
# Sentence boundary helpers
# ---------------------------------------------------------------------------

def mention_sentence(span: List[int], sentences: List[List[int]]) -> int:
    """Return the 0-based sentence index that contains *span*'s first token."""
    if not span:
        return -1
    tok = span[0]
    for idx, (s, e) in enumerate(sentences):
        if s <= tok < e:
            return idx
    return len(sentences)  # past last known sentence


# ---------------------------------------------------------------------------
# Dataset loading and indexing
# ---------------------------------------------------------------------------

# Cache: (repo_id, split) → {doc_id: {mention_id: sent_idx}}
_DATASET_CACHE: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = {}


def load_doc_index(repo_id: str, split: str) -> Dict[str, Dict[str, int]]:
    """Load an HF dataset split and return {doc_id: {mention_id: sent_idx}}."""
    key = (repo_id, split)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    ds = load_dataset(repo_id, split=split)
    index: Dict[str, Dict[str, int]] = {}
    for row in ds:
        doc_id: str = row["id"]
        mentions: List[str] = row["mentions"]
        spans: List[List[int]] = row["spans"]
        sentences: List[List[int]] = row["sentences"]

        index[doc_id] = {
            m: mention_sentence(sp, sentences)
            for m, sp in zip(mentions, spans)
        }

    _DATASET_CACHE[key] = index
    return index


def dataset_key_to_config(ds_key: str, datasets_cfg: dict) -> Optional[Tuple[str, str]]:
    """Extract (repo_id, split) from a run's config dict."""
    ds_cfg = datasets_cfg.get(ds_key)
    if not ds_cfg:
        return None
    repo_id = ds_cfg.get("repo_id")
    split = ds_cfg.get("split", "test")
    if not repo_id:
        return None
    return repo_id, split


# ---------------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------------

def classify_pair(pair_str: str, mention_to_sent: Dict[str, int]) -> Optional[str]:
    """Return 'intra', 'inter', or None if mentions are unknown."""
    parts = pair_str.split(",", 1)
    if len(parts) != 2:
        return None
    s1 = mention_to_sent.get(parts[0].strip())
    s2 = mention_to_sent.get(parts[1].strip())
    if s1 is None or s2 is None:
        return None
    return "intra" if s1 == s2 else "inter"


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyze_run(log_path: Path) -> Optional[Dict]:
    with open(log_path) as f:
        data = json.load(f)

    config = data.get("config") or {}
    active_ds = config.get("active_dataset")
    if not active_ds:
        return None

    datasets_cfg = config.get("datasets") or {}
    ds_info = dataset_key_to_config(active_ds, datasets_cfg)
    if not ds_info:
        return None

    repo_id, split = ds_info
    try:
        doc_index = load_doc_index(repo_id, split)
    except Exception as exc:
        print(f"  Skipping {log_path.name}: could not load dataset — {exc}", file=sys.stderr)
        return None

    per_pair = (data.get("results") or {}).get("per_pair_predictions") or []

    intra: List[Dict] = []
    inter: List[Dict] = []
    unclassified = 0

    for pred in per_pair:
        if pred.get("gold") == "unannotated":
            continue
        mention_to_sent = doc_index.get(pred["id"], {})
        kind = classify_pair(pred["pair"], mention_to_sent)
        if kind == "intra":
            intra.append(pred)
        elif kind == "inter":
            inter.append(pred)
        else:
            unclassified += 1

    run_id: str = data.get("run_id", log_path.stem)
    model: str = (config.get("model") or {}).get("default_model_id", "unknown")
    timestamp: str = data.get("timestamp_utc", "")[:19]

    row: Dict = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": active_ds,
        "model": model,
        "total_pairs": len(per_pair),
        "intra_count": len(intra),
        "inter_count": len(inter),
        "unclassified": unclassified,
    }

    for label, pairs in [("intra", intra), ("inter", inter)]:
        if pairs:
            y_true = [p["gold"] for p in pairs]
            y_pred = [p["pred"] for p in pairs]
            m = compute_binary_metrics(y_true, y_pred)
            row[f"{label}_f1"] = round(m["f1"], 4)
            row[f"{label}_precision"] = round(m["precision"], 4)
            row[f"{label}_recall"] = round(m["recall"], 4)
            row[f"{label}_support_pos"] = m["support_pos"]
        else:
            for metric in ("f1", "precision", "recall", "support_pos"):
                row[f"{label}_{metric}"] = None

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare intra vs inter-sentence binary metrics across run logs."
    )
    parser.add_argument("--logs", default=str(LOGS_DIR), help="Log directory (default: logs/allatonce)")
    parser.add_argument("--dataset", help="Restrict to one dataset key (e.g. meci, maven_ere)")
    parser.add_argument("--run_ids", nargs="*", help="Restrict to specific run IDs")
    parser.add_argument("--out", help="Write results to this CSV path")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    log_files = sorted(log_dir.glob("run_*.json"))
    if not log_files:
        sys.exit(f"No run_*.json files found in {log_dir}")

    print(f"Found {len(log_files)} log file(s). Analysing…\n")

    rows = []
    for log_path in log_files:
        result = analyze_run(log_path)
        if result is None:
            continue
        if args.dataset and result["dataset"] != args.dataset:
            continue
        if args.run_ids and result["run_id"] not in args.run_ids:
            continue
        rows.append(result)

    if not rows:
        sys.exit("No results after filtering.")

    df = pd.DataFrame(rows)

    col_order = [
        "run_id", "timestamp", "dataset", "model",
        "total_pairs", "intra_count", "inter_count", "unclassified",
        "intra_f1", "intra_precision", "intra_recall", "intra_support_pos",
        "inter_f1", "inter_precision", "inter_recall", "inter_support_pos",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix == ".md":
            out_path.write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")
        else:
            df.to_csv(out_path, index=False)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()


# uv run python scripts/analysis/pair_cot_errors.py --latest logs/allatonce/ --n-pairs 3