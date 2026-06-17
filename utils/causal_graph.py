"""
Shared Graphviz rendering for TP/FP/FN causal-relation graphs, plus live mlflow
logging. Used both by the standalone scripts/analysis/causal_graph.py (log-file
based) and by the per-document mlflow logging in main.py (triple-set based).
"""
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile

import graphviz

from utils.labels import CAUSED_BY, NOREL_VARIANTS

TP_COLOR = "#1a9850"  # green: gold causal, predicted causal
FP_COLOR = "#d73027"  # red:   gold norel,  predicted causal
FN_COLOR = "#4575b4"  # blue:  gold causal, predicted norel

_COLOR_BY_STATUS = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}


def is_causal(label: str) -> bool:
    return str(label).lower() not in NOREL_VARIANTS


def build_edges_from_triples(gold_triples, pred_triples) -> list[dict]:
    """Classify (src, label, tgt) relation triples into TP / FP / FN edges,
    directed cause -> effect. Mirrors the set logic of compute_ere_metrics."""
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


def build_graph(doc_id: str, mentions_map: dict[str, str], edges: list[dict], fmt: str = "png") -> graphviz.Digraph:
    """Builds the Graphviz Digraph object (nodes = mentions, edges = TP/FP/FN) without rendering."""
    g = graphviz.Digraph("causal_graph", format=fmt)
    g.attr(rankdir="LR", label=doc_id, labelloc="t", fontsize="16")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#fffde7", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10")

    used_mentions = {m for e in edges for m in (e["src"], e["tgt"])}
    nodes_to_draw = used_mentions if used_mentions else set(mentions_map)
    for mention_id in sorted(nodes_to_draw, key=lambda m: (len(m), m)):
        text = mentions_map.get(mention_id, mention_id)
        g.node(mention_id, label=f"{mention_id}\n{text}")

    for e in edges:
        g.edge(e["src"], e["tgt"], color=_COLOR_BY_STATUS[e["status"]], penwidth="2", label=e["label"])

    with g.subgraph(name="legend") as lg:
        lg.attr(rank="sink")
        lg.attr("node", shape="plaintext", style="", fontsize="10")
        lg.node("legend_label", label="Legend:")
        lg.node("legend_tp", label="TP (correct causal)", fontcolor=TP_COLOR)
        lg.node("legend_fp", label="FP (hallucinated causal)", fontcolor=FP_COLOR)
        lg.node("legend_fn", label="FN (missed causal)", fontcolor=FN_COLOR)

    return g


def render_graph(doc_id: str, mentions_map: dict[str, str], edges: list[dict], out_path: str, fmt: str = "png") -> str:
    """Renders nodes (mention id + text) and TP/FP/FN edges to out_path.<fmt>. Returns the rendered file path."""
    g = build_graph(doc_id, mentions_map, edges, fmt=fmt)
    return g.render(out_path, cleanup=True)


def safe_filename(doc_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", doc_id)


def log_to_mlflow(
    doc_id: str,
    doc_idx: int,
    mentions_map: dict[str, str],
    gold_triples,
    pred_triples,
    trace_ids: list[str],
) -> None:
    """Renders the TP/FP/FN graph for one document and logs it to the active mlflow run:
    - an SVG artifact under causal_graphs/<doc_id>.svg (small, vector, previewable in the UI)
    - a PNG via mlflow.log_image (populates the run's Images gallery, scrubbable by doc_idx)
    Also tags each of the document's trace ids with the artifact path for cross-reference.
    Never raises — a rendering/logging failure must not fail the document's actual result.
    """
    try:
        import mlflow
        from PIL import Image

        edges = build_edges_from_triples(gold_triples, pred_triples)
        safe_id = safe_filename(doc_id)

        g = build_graph(doc_id, mentions_map, edges)

        tmp_dir = tempfile.mkdtemp(prefix="causal_graph_")
        try:
            svg_path = g.render(os.path.join(tmp_dir, safe_id), format="svg", cleanup=True)
            mlflow.log_artifact(svg_path, artifact_path="causal_graphs")

            png_bytes = g.pipe(format="png")
            img = Image.open(io.BytesIO(png_bytes))
            mlflow.log_image(img, key="causal_graph", step=doc_idx)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        artifact_path = f"causal_graphs/{safe_id}.svg"
        for req_id in trace_ids:
            if req_id:
                mlflow.set_trace_tag(req_id, "causal_graph_artifact", artifact_path)
    except Exception as e:
        print(f"[causal_graph] Warning: failed to log causal graph for {doc_id}: {e}")
