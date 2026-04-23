#!/usr/bin/env python3
"""
undirected_binary_eval.py

Loads an experiment log JSON and computes direction-agnostic binary metrics.

All non-NoRel labels are collapsed to "Causal".
Two undirected aggregation modes:
  - OR  (lenient):      pair is Causal if EITHER direction is Causal
  - AND (conservative):  pair is Causal only if BOTH directions are Causal
                          (if only one direction exists, that single vote stands)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def binarize(label: str) -> str:
    """Collapse any relation to 'Causal'; keep 'NoRel' as-is."""
    return "NoRel" if str(label).lower() in ("norel", "none", "unknown", "") else "Causal"


def undirected_key(src: str, tgt: str) -> Tuple[str, str]:
    """Canonical (sorted) undirected pair key."""
    return tuple(sorted([src, tgt]))


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ═════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def build_undirected_pairs(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[Tuple[str, str], Dict[str, List[str]]]]:
    """
    Groups directed pair predictions into undirected pairs, per document.

    Returns:
        { doc_id: { (A, B): {"gold": [lbl, ...], "pred": [lbl, ...]} } }

    Each undirected pair can have up to two entries (one per direction).
    """
    docs: Dict[str, Dict[Tuple[str, str], Dict[str, List[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"gold": [], "pred": []})
    )

    for row in rows:
        doc_id = row["id"]
        pair_str = row["pair"]
        if "," not in pair_str:
            continue
        src, tgt = [x.strip() for x in pair_str.split(",", 1)]
        ukey = undirected_key(src, tgt)

        docs[doc_id][ukey]["gold"].append(binarize(row["gold"]))
        docs[doc_id][ukey]["pred"].append(binarize(row["pred"]))

    return docs


def aggregate_pair(labels: List[str], mode: str) -> str:
    """
    Reduces one or two directed binary labels into a single undirected label.

    Args:
        labels: 1 or 2 binarized labels (one per direction that exists).
        mode:   'or' (lenient) | 'and' (conservative)
    """
    if not labels:
        return "NoRel"
    if len(labels) == 1:
        return labels[0]
    # len >= 2
    if mode == "or":
        return "Causal" if any(l == "Causal" for l in labels) else "NoRel"
    else:  # "and"
        return "Causal" if all(l == "Causal" for l in labels) else "NoRel"


def evaluate(
    docs: Dict[str, Dict[Tuple[str, str], Dict[str, List[str]]]],
    mode: str,
) -> Dict[str, Any]:
    """
    Runs evaluation for a given aggregation mode.

    Returns global + per-language + per-document metrics.
    """
    global_tp = global_fp = global_fn = global_tn = 0
    per_lang: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    per_doc_results: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    # We need lang info – we'll grab it from the original rows in the caller
    # and attach it here.  For simplicity, we pass it through the docs dict.
    # Actually, the rows already carry 'lang'. Let's restructure slightly.
    # ---> Handled by the caller attaching lang to each doc via a separate map.

    for doc_id, pairs in docs.items():
        doc_tp = doc_fp = doc_fn = doc_tn = 0

        for ukey, signals in pairs.items():
            g = aggregate_pair(signals["gold"], mode)
            p = aggregate_pair(signals["pred"], mode)

            if g == "Causal" and p == "Causal":
                doc_tp += 1
            elif g == "NoRel" and p == "Causal":
                doc_fp += 1
            elif g == "Causal" and p == "NoRel":
                doc_fn += 1
            else:
                doc_tn += 1

            detail_rows.append({
                "doc_id": doc_id,
                "pair": f"{ukey[0]},{ukey[1]}",
                "gold_undirected": g,
                "pred_undirected": p,
                "gold_directed": signals["gold"],
                "pred_directed": signals["pred"],
            })

        global_tp += doc_tp
        global_fp += doc_fp
        global_fn += doc_fn
        global_tn += doc_tn

        per_doc_results.append({
            "doc_id": doc_id,
            "pairs": doc_tp + doc_fp + doc_fn + doc_tn,
            **precision_recall_f1(doc_tp, doc_fp, doc_fn),
        })

    global_metrics = {
        **precision_recall_f1(global_tp, global_fp, global_fn),
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "tn": global_tn,
        "total_pairs": global_tp + global_fp + global_fn + global_tn,
    }

    return {
        "global": global_metrics,
        "per_doc": per_doc_results,
        "detail_rows": detail_rows,
    }


def evaluate_per_lang(
    rows: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Runs undirected binary eval grouped by language."""
    lang_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        lang_rows[r.get("lang", "unknown")].append(r)

    results = {}
    for lang, lr in lang_rows.items():
        docs = build_undirected_pairs(lr)
        ev = evaluate(docs, mode)
        results[lang] = ev["global"]
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Undirected binary evaluation of a logged experiment run."
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to the run JSON log file (e.g. logs/run_2025-...json)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the full report JSON.",
    )
    args = parser.parse_args()

    # 1. Load log
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    rows = log_data.get("results", {}).get("per_pair_predictions", [])
    if not rows:
        print("Error: no 'results.per_pair_predictions' found in log.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} directed pair predictions from {log_path.name}\n")

    # 2. Evaluate under both modes
    report = {}
    for mode in ("or", "and"):
        docs = build_undirected_pairs(rows)
        ev = evaluate(docs, mode)
        lang_ev = evaluate_per_lang(rows, mode)

        report[mode] = {
            "global": ev["global"],
            "per_lang": lang_ev,
            "per_doc": ev["per_doc"],
            "detail_rows": ev["detail_rows"],
        }

        label = "OR  (lenient)    " if mode == "or" else "AND (conservative)"
        g = ev["global"]
        print(f"── {label} ──")
        print(f"   Precision : {g['precision']:.4f}")
        print(f"   Recall    : {g['recall']:.4f}")
        print(f"   F1        : {g['f1']:.4f}")
        print(f"   TP={g['tp']}  FP={g['fp']}  FN={g['fn']}  TN={g['tn']}  "
              f"Total undirected pairs={g['total_pairs']}")

        if lang_ev:
            print(f"   Per-language:")
            for lang, lm in sorted(lang_ev.items()):
                print(f"     {lang:>6s}:  P={lm['precision']:.4f}  "
                      f"R={lm['recall']:.4f}  F1={lm['f1']:.4f}  "
                      f"(pairs={lm['total_pairs']})")
        print()

    # 3. Optionally save
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Full report saved to {out_path}")


if __name__ == "__main__":
    main()