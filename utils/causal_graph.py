"""
Shared Graphviz rendering for TP/FP/FN causal-relation graphs, plus live mlflow
logging. Used both by the standalone scripts/analysis/causal_graph.py (log-file
based) and by the per-document mlflow logging in main.py (triple-set based).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import textwrap

import graphviz

from utils.labels import CAUSED_BY, NOREL_VARIANTS

TP_COLOR = "#009E73"  # green:  gold causal, predicted causal (Okabe-Ito)
FP_COLOR = "#E69F00"  # orange: gold norel,  predicted causal (Okabe-Ito)
FN_COLOR = "#0072B2"  # blue:   gold causal, predicted norel  (Okabe-Ito)

_COLOR_BY_STATUS = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}
_STYLE_BY_STATUS = {"TP": "solid", "FP": "dotted", "FN": "dashed"}


def is_causal(label: str) -> bool:
    return str(label).lower() not in NOREL_VARIANTS


def build_edges_from_triples(gold_triples, pred_triples, include_norel: bool = False) -> list[dict]:
    """Classify (src, label, tgt) relation triples into TP / FP / FN edges,
    directed cause -> effect. Mirrors the set logic of compute_ere_metrics.
    """

    gold_set = set(gold_triples)
    pred_set = set(pred_triples)

    def to_edge(triple: tuple, status: str) -> dict:
        src, label, tgt = triple
        if str(label).lower() == CAUSED_BY:
            cause, effect = tgt, src
        else:
            cause, effect = src, tgt
        return {"src": cause, "tgt": effect, "status": status, "label": label}

    edges = [to_edge(t, "TP") for t in gold_set & pred_set]
    edges += [to_edge(t, "FP") for t in pred_set - gold_set]
    edges += [to_edge(t, "FN") for t in gold_set - pred_set]
    return edges


def build_graph(doc_id: str, mentions_map: dict[str, str], edges: list[dict], fmt: str = "png", doc_text: str | None = None) -> graphviz.Digraph:
    """Builds the Graphviz Digraph object (nodes = mentions, edges = TP/FP/FN) without rendering."""
    g = graphviz.Digraph("causal_graph", format=fmt)
    g.attr(rankdir="LR", label=doc_id, labelloc="t", fontsize="16", pad="0.4,0.2")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#fffde7", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10")

    if doc_text:
        with g.subgraph(name="doc_text") as dt:
            dt.attr(rank="source")
            dt.node("doc_text", shape="note", style="filled", fillcolor="white", fontsize="10",
                     label=textwrap.fill(doc_text, width=100))

    used_mentions = {m for e in edges for m in (e["src"], e["tgt"])}
    nodes_to_draw = used_mentions if used_mentions else set(mentions_map)
    for mention_id in sorted(nodes_to_draw, key=lambda m: (len(m), m)):
        text = mentions_map.get(mention_id, mention_id)
        g.node(mention_id, label=f"{mention_id}\n{text}")

    for e in edges:
        g.edge(e["src"], e["tgt"], color=_COLOR_BY_STATUS[e["status"]],
               style=_STYLE_BY_STATUS[e["status"]], penwidth="2", label=e["label"])

    with g.subgraph(name="legend") as lg:
        lg.attr(rank="sink")
        lg.attr("node", shape="point", style="invis")
        lg.attr("edge", fontname="Helvetica", fontsize="10", penwidth="2")
        lg.node("legend_label", shape="plaintext", style="", label="Legend:")
        for status, text in (("TP", "correct causal"), ("FP", "hallucinated causal"), ("FN", "missed causal")):
            lg.node(f"legend_{status}_a")
            lg.node(f"legend_{status}_b")
            lg.edge(f"legend_{status}_a", f"legend_{status}_b",
                    color=_COLOR_BY_STATUS[status], style=_STYLE_BY_STATUS[status],
                    label=f"{status} ({text})", fontcolor=_COLOR_BY_STATUS[status])

    return g


def render_graph(doc_id: str, mentions_map: dict[str, str], edges: list[dict], out_path: str, fmt: str = "png", doc_text: str | None = None) -> str:
    """Renders nodes (mention id + text) and TP/FP/FN edges to out_path.<fmt>. Returns the rendered file path."""
    g = build_graph(doc_id, mentions_map, edges, fmt=fmt, doc_text=doc_text)
    return g.render(out_path, cleanup=True)


def safe_filename(doc_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", doc_id)


def log_to_mlflow(
    doc_id: str,
    mentions_map: dict[str, str],
    gold_triples,
    pred_triples,
    trace_ids: list[str],
    doc_text: str | None = None,
) -> None:
    """Renders the TP/FP/FN graph for one document and logs it as an SVG artifact
    under causal_graphs/<doc_id>.svg on the active mlflow run (small, vector,
    previewable directly in the Run's Artifacts tab). Also tags each of the
    document's trace ids with the artifact path for cross-reference.
    Never raises — a rendering/logging failure must not fail the document's actual result.
    """
    try:
        import mlflow

        edges = build_edges_from_triples(gold_triples, pred_triples)
        safe_id = safe_filename(doc_id)

        g = build_graph(doc_id, mentions_map, edges, doc_text=doc_text)

        tmp_dir = tempfile.mkdtemp(prefix="causal_graph_")
        try:
            svg_path = g.render(os.path.join(tmp_dir, safe_id), format="svg", cleanup=True)
            mlflow.log_artifact(svg_path, artifact_path="causal_graphs")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        artifact_path = f"causal_graphs/{safe_id}.svg"
        for req_id in trace_ids:
            if req_id:
                mlflow.set_trace_tag(req_id, "causal_graph_artifact", artifact_path)
    except Exception as e:
        print(f"[causal_graph] Warning: failed to log causal graph for {doc_id}: {e}")
