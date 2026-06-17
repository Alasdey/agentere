#!/usr/bin/env python3
"""
Render a Graphviz graph of predicted vs. gold causal relations for one document.

Nodes are event mentions (id + mention text) shown as bubbles. Edges connect
mention pairs that are causal in gold and/or in the prediction, colored by
outcome:
  - TP (green):  gold causal, predicted causal
  - FP (red):    gold norel,  predicted causal
  - FN (blue):   gold causal, predicted norel
True negatives (gold norel, pred norel) are not drawn, to keep the graph readable.

Joins against the source HuggingFace dataset to resolve mention text, the
same way scripts/analysis/pair_errors.py does.

Usage:
    uv run scripts/analysis/causal_graph.py logs/allatonce/run_XYZ.json --doc-id some_doc.ann
    uv run scripts/analysis/causal_graph.py --latest logs/allatonce/ --doc-id some_doc.ann
    uv run scripts/analysis/causal_graph.py logs/allatonce/run_*.json --doc-id some_doc.ann --out figs/some_doc
    uv run scripts/analysis/causal_graph.py --latest logs/allatonce/ --list-docs
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import graphviz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import json
from utils.labels import CAUSES, CAUSED_BY, NOREL_VARIANTS  # noqa: E402

TP_COLOR = "#1a9850"  # green
FP_COLOR = "#d73027"  # red
FN_COLOR = "#4575b4"  # blue


def is_causal(label: str) -> bool:
    return str(label).lower() not in NOREL_VARIANTS


def build_mentions_map(row: dict) -> dict[str, str]:
    tokens = row["tokens"]
    spans = row["spans"]
    mentions = row["mentions"]
    result = {}
    for j, mention_id in enumerate(mentions):
        parts = [tokens[idx] for idx in spans[j] if 0 <= idx < len(tokens)]
        result[str(mention_id)] = " ".join(parts) if parts else str(mention_id)
    return result


def get_ds_key(log_path: str) -> tuple[str, str]:
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["config"]
    active_ds = cfg["active_dataset"]
    ds_cfg = cfg["datasets"][active_ds]
    return (ds_cfg["repo_id"], ds_cfg["split"])


def load_doc_rows(repo_id: str, split: str) -> dict[str, dict]:
    """Returns {doc_id: dataset_row} for the full split."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split)
    return {str(row["id"]): row for row in ds}


def collect_pairs(log_paths: list[str], doc_id: str) -> dict[tuple[str, str], dict]:
    """Aggregate predictions for one doc across log files. Majority vote per ordered pair."""
    votes: dict[tuple[str, str], dict] = defaultdict(lambda: {"gold": None, "pred_counts": defaultdict(int)})
    for log_path in log_paths:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data["results"]["per_pair_predictions"]:
            if str(row["id"]) != doc_id:
                continue
            src, tgt = [p.strip() for p in row["pair"].split(",", 1)]
            key = (src, tgt)
            entry = votes[key]
            entry["gold"] = row["gold"]
            entry["pred_counts"][row["pred"]] += 1
    pairs = {}
    for key, entry in votes.items():
        majority_pred = max(entry["pred_counts"], key=entry["pred_counts"].__getitem__)
        pairs[key] = {"gold": entry["gold"], "pred": majority_pred}
    return pairs


def build_edges(pairs: dict[tuple[str, str], dict]) -> list[dict]:
    """Collapse mirrored ordered pairs into one directed edge per unordered pair,
    classified as TP / FP / FN. True negatives are dropped."""
    seen: set[frozenset] = set()
    edges = []
    for (src, tgt), info in pairs.items():
        unordered = frozenset((src, tgt))
        if unordered in seen:
            continue

        gold, pred = info["gold"], info["pred"]
        mirror = pairs.get((tgt, src))

        if is_causal(gold):
            # canonical direction: "causes" label points cause -> effect
            if gold.lower() == CAUSES:
                cause, effect, fwd_pred = src, tgt, pred
            elif gold.lower() == CAUSED_BY:
                cause, effect, fwd_pred = tgt, src, mirror["pred"] if mirror else pred
            else:
                cause, effect, fwd_pred = src, tgt, pred
            seen.add(unordered)

            if is_causal(fwd_pred):
                edges.append({"src": cause, "tgt": effect, "status": "TP", "label": fwd_pred})
            else:
                edges.append({"src": cause, "tgt": effect, "status": "FN", "label": gold})
        else:
            seen.add(unordered)
            if is_causal(pred):
                if pred.lower() == CAUSED_BY:
                    edges.append({"src": tgt, "tgt": src, "status": "FP", "label": pred})
                else:
                    edges.append({"src": src, "tgt": tgt, "status": "FP", "label": pred})
            # else: true negative, dropped

    return edges


def render_graph(doc_id: str, mentions_map: dict[str, str], edges: list[dict], out_path: str, fmt: str):
    g = graphviz.Digraph("causal_graph", format=fmt)
    g.attr(rankdir="LR", label=doc_id, labelloc="t", fontsize="16")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#fffde7", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10")

    used_mentions = {m for e in edges for m in (e["src"], e["tgt"])}
    nodes_to_draw = used_mentions if used_mentions else set(mentions_map)
    for mention_id in sorted(nodes_to_draw, key=lambda m: (len(m), m)):
        text = mentions_map.get(mention_id, mention_id)
        g.node(mention_id, label=f"{mention_id}\n{text}")

    color_by_status = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}
    for e in edges:
        g.edge(e["src"], e["tgt"], color=color_by_status[e["status"]],
               penwidth="2", label=e["label"])

    with g.subgraph(name="legend") as lg:
        lg.attr(rank="sink")
        lg.attr("node", shape="plaintext", style="", fontsize="10")
        lg.node("legend_label", label="Legend:")
        lg.node("legend_tp", label="TP (correct causal)", fontcolor=TP_COLOR)
        lg.node("legend_fp", label="FP (hallucinated causal)", fontcolor=FP_COLOR)
        lg.node("legend_fn", label="FN (missed causal)", fontcolor=FN_COLOR)

    rendered_path = g.render(out_path, cleanup=True)
    return rendered_path


def main():
    parser = argparse.ArgumentParser(description="Graphviz visualization of TP/FP/FN causal predictions for one document")
    parser.add_argument("logs", nargs="*", help="Log JSON file(s) or directory")
    parser.add_argument("--latest", action="store_true", help="Use only the most recent log in the given dir/glob")
    parser.add_argument("--doc-id", help="Document id to render")
    parser.add_argument("--list-docs", action="store_true", help="List available doc ids in the given log(s) and exit")
    parser.add_argument("--out", help="Output path without extension (default: figs/causal_graph_<doc_id>)")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"], help="Output format")
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
    if args.latest:
        paths = [paths[-1]]

    if args.list_docs:
        doc_ids = set()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            for row in data["results"]["per_pair_predictions"]:
                doc_ids.add(str(row["id"]))
        for d in sorted(doc_ids):
            print(d)
        return

    if not args.doc_id:
        print("Error: --doc-id is required (use --list-docs to see options)", file=sys.stderr)
        sys.exit(1)

    repo_id, split = get_ds_key(paths[0])
    print(f"Loading dataset {repo_id} ({split})...")
    doc_rows = load_doc_rows(repo_id, split)
    if args.doc_id not in doc_rows:
        print(f"Error: doc_id '{args.doc_id}' not found in dataset.", file=sys.stderr)
        sys.exit(1)
    mentions_map = build_mentions_map(doc_rows[args.doc_id])

    pairs = collect_pairs(paths, args.doc_id)
    if not pairs:
        print(f"Error: no predictions found for doc_id '{args.doc_id}' in given log(s).", file=sys.stderr)
        sys.exit(1)

    edges = build_edges(pairs)
    tp = sum(1 for e in edges if e["status"] == "TP")
    fp = sum(1 for e in edges if e["status"] == "FP")
    fn = sum(1 for e in edges if e["status"] == "FN")
    print(f"  Edges: TP={tp} FP={fp} FN={fn}")

    out_path = args.out or os.path.join("figs", f"causal_graph_{args.doc_id}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rendered = render_graph(args.doc_id, mentions_map, edges, out_path, args.format)
    print(f"  Written to: {rendered}")


if __name__ == "__main__":
    main()
