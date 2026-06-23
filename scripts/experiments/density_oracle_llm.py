#!/usr/bin/env python3
"""
density_oracle_llm.py
~~~~~~~~~~~~~~~~~~~~~~
Scale-up of the manual blind-subagent experiment: given a BATCH of
documents (mentions tagged inline as `<ID text>`), ask an LLM to read all
of them and rank them from lowest to highest causal density (`n_relations
/ n_mentions`), WITHOUT access to gold labels. This mirrors how the manual
test actually worked — relative judgment across a handful of documents
seen side by side, not an isolated absolute guess with no calibration
reference (which collapsed to a near-constant answer in practice).

Each dataset's train split is sampled (no filtering — every document is
eligible) into batches of `--batch-size` documents that are PERCENTILE-
BALANCED: every single batch contains one document from each density
bin (lowest bin through highest bin), so every batch has the same kind of
low-to-high contrast the manual test got by hand-picking specific
percentiles — a random shuffle-then-chunk could otherwise land an entire
batch on mid-range documents, making ranking meaningless. One concurrent
LLM call per batch. We score:
  - the batch's stated low->high ranking against the true ranking
    (Spearman over rank positions — the primary metric, comparable to the
    manual subagent test's "ranking (low to high): ..." score)
  - the per-document absolute `est_relations` guesses the model also gives
    us "for free" inside the same JSON, against real R/density (secondary,
    pointwise metric), plus a naive mean-density baseline for comparison.

Usage
-----
    uv run scripts/experiments/density_oracle_llm.py \\
        --model "qwen/qwen3-30b-a3b-instruct-2507" \\
        --datasets meci,maven_ere,causal_timebank,event_story_line \\
        --n-docs 28 --batch-size 4

Run with --help for all options. Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

DATASET_KEYS = ["meci", "maven_ere", "causal_timebank", "event_story_line"]

SYSTEM_PROMPT = (
    "You are triaging documents for an Event Causality Identification (ECI) "
    "pipeline before deciding how much effort to spend on full pairwise "
    "extraction. You are given SEVERAL documents, each labeled and each "
    "with event mentions tagged inline as <ID text> (e.g. <e3 collapsed>).\n\n"
    "For EACH document, estimate how many CAUSAL relation pairs exist among "
    "its tagged mentions — a pair counts only if one event directly "
    "causes/enables/leads to the other. Do not count: shared topic, mere "
    "temporal sequence, or multi-step chains flattened into one direct link "
    "(e.g. 'A leads to B leads to C' is two relations, not a direct A-C "
    "link). Do not exhaustively enumerate every pair in writing — just name "
    "the causal chains that stand out on a single read-through of each "
    "document, the same way you'd skim-read it to triage it.\n\n"
    "Then RANK all the documents from LOWEST to HIGHEST causal density "
    "(density = causal relations / mentions). Use the relative comparison "
    "across documents you were just given — that comparison is the point "
    "of seeing them together, not an isolated absolute judgment per doc.\n\n"
    "After your reasoning, end with ONE final JSON object (no markdown "
    "fence) on its own line:\n"
    '{"estimates": [{"label": "A", "n_mentions_seen": <int>, '
    '"est_relations": <int>}, ...], '
    '"ranking_low_to_high": ["<label>", "<label>", ...]}\n\n'
    "est_relations must be a non-negative integer (0, 1, 2, ...) — use 0 "
    "for none, never a negative number. ranking_low_to_high must list "
    "EVERY label given to you exactly once, ordered lowest density first."
)


def _batch_user_prompt(labels: List[str], docs: List[Dict[str, Any]]) -> str:
    parts = [f"You are given {len(docs)} documents: {', '.join(labels)}.\n"]
    for label, doc in zip(labels, docs):
        parts.append(f"=== Document {label} ===\n{doc['text']}\n")
    parts.append("Analyze each document, then give your final JSON answer.")
    return "\n".join(parts)


def load_repo_map() -> Dict[str, Dict[str, str]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: cfg["datasets"][k] for k in DATASET_KEYS}


def load_doc_pool(ds_key: str, repo_id: str) -> List[Dict[str, Any]]:
    """Load the full train split and compute (text, M, R, density) per document.

    No filtering by mention count, language, or anything else — every
    document in the split is eligible for sampling, so the density range
    seen by the model matches the corpus's true range exactly.
    """
    ds = load_dataset(repo_id, split="train")
    pool = []
    for i, row in enumerate(ds):
        mentions = row.get("mentions", [])
        M = len(mentions)
        if M == 0:
            continue  # density (R/M) is undefined, not a filtering choice
        relations = row.get("relations", {}) or {}
        pair_set = set()
        for k in relations:
            if k.lower() == "norel":
                continue
            for a, b in relations[k]:
                pair_set.add(frozenset((a, b)))
        R = len(pair_set)
        pool.append({
            "row_idx": i,
            "id": str(row.get("id", f"idx_{i}")),
            "text": row.get("text", "") or "",
            "M": M,
            "R": R,
            "density": R / M,
        })
    return pool


def _evenly_spaced_indices(length: int, count: int, seed: int) -> List[int]:
    """count indices spread evenly across range(length), seeded for ties/padding."""
    if length <= 0:
        return []
    if count >= length:
        return [i % length for i in range(count)]
    idxs = sorted(set(int(round(p)) for p in np.linspace(0, length - 1, count)))
    rng = random.Random(seed)
    while len(idxs) < count:
        candidate = rng.randrange(length)
        if candidate not in idxs:
            idxs.append(candidate)
    return sorted(idxs)[:count]


def build_percentile_balanced_batches(
    pool: List[Dict[str, Any]], n_docs: int, batch_size: int, seed: int
) -> List[List[Dict[str, Any]]]:
    """Build batches where EVERY batch spans the full density distribution.

    The pool is sorted by density and sliced into `batch_size` percentile
    bins (lowest bin, ..., highest bin). Each batch draws exactly one
    document from each bin, evenly spaced within that bin across batches —
    so every single batch contains a genuine low-density doc through a
    genuine high-density doc, the same contrast the manual subagent test
    got by hand-picking specific percentiles. A random shuffle-then-chunk
    (the previous approach) could not guarantee this: a batch could land
    entirely on mid-range documents, making ranking meaningless.
    """
    n_batches = max(1, n_docs // batch_size)
    pool_sorted = sorted(pool, key=lambda d: d["density"])
    n = len(pool_sorted)

    bin_bounds = [int(round(b * n / batch_size)) for b in range(batch_size + 1)]
    per_bin_docs: List[List[Dict[str, Any]]] = []
    for b in range(batch_size):
        bin_pool = pool_sorted[bin_bounds[b]:bin_bounds[b + 1]] or pool_sorted
        local_idxs = _evenly_spaced_indices(len(bin_pool), n_batches, seed + b)
        per_bin_docs.append([bin_pool[i] for i in local_idxs])

    rng = random.Random(seed)
    batches = []
    for i in range(n_batches):
        batch = [per_bin_docs[b][i] for b in range(batch_size)]
        rng.shuffle(batch)  # don't let label order leak the density order
        batches.append(batch)
    return batches


def extract_balanced_json_objects(text: str) -> List[str]:
    """Return every brace-balanced top-level {...} substring in text."""
    results, stack, start = [], [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    results.append(text[start:i + 1])
                    start = None
    return results


def parse_batch_response(raw: str, labels: List[str]) -> Optional[Dict[str, Any]]:
    candidates = [c for c in extract_balanced_json_objects(raw) if "ranking_low_to_high" in c]
    label_set = set(labels)
    for candidate in reversed(candidates):  # prefer the LAST one (final answer, after reasoning)
        try:
            obj = json.loads(candidate)
            ranking = list(obj["ranking_low_to_high"])
            if set(ranking) != label_set or len(ranking) != len(labels):
                continue
            estimates: Dict[str, Optional[int]] = {label: None for label in labels}
            for item in obj.get("estimates", []):
                label = item.get("label")
                if label not in label_set:
                    continue
                try:
                    estimates[label] = max(0, int(item["est_relations"]))
                except Exception:
                    pass
            return {"ranking": ranking, "estimates": estimates}
        except Exception:
            continue
    return None


def batch_rank_spearman(true_order: List[str], model_order: List[str]) -> float:
    true_rank = {label: i for i, label in enumerate(true_order)}
    model_rank = {label: i for i, label in enumerate(model_order)}
    x = np.array([true_rank[l] for l in true_order], dtype=float)
    y = np.array([model_rank[l] for l in true_order], dtype=float)
    return spearman(x, y)


async def run_batch(
    llm: ChatOpenAI, batch_docs: List[Dict[str, Any]], retries: int, semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    labels = list(string.ascii_uppercase[:len(batch_docs)])
    docs_by_label = dict(zip(labels, batch_docs))
    true_order = [label for label, _ in sorted(docs_by_label.items(), key=lambda kv: kv[1]["density"])]

    async with semaphore:
        last_err = None
        for attempt in range(retries + 1):
            try:
                response = await llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_batch_user_prompt(labels, batch_docs)),
                ])
                parsed = parse_batch_response(response.content, labels)
                if parsed is not None:
                    per_doc = []
                    for label, doc in docs_by_label.items():
                        est = parsed["estimates"].get(label)
                        per_doc.append({
                            "id": doc["id"], "label": label, "M": doc["M"], "R_true": doc["R"],
                            "density_true": doc["density"], "R_pred": est,
                            "density_pred": (est / doc["M"]) if est is not None else None,
                        })
                    return {
                        "ok": True,
                        "rank_spearman": batch_rank_spearman(true_order, parsed["ranking"]),
                        "true_order": true_order, "model_order": parsed["ranking"],
                        "per_doc": per_doc, "raw": response.content,
                    }
                last_err = f"unparseable/invalid batch response: {response.content[:400]!r}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(1.5 * (attempt + 1))

        return {
            "ok": False, "rank_spearman": float("nan"), "true_order": true_order, "model_order": None,
            "per_doc": [
                {"id": d["id"], "label": label, "M": d["M"], "R_true": d["R"],
                 "density_true": d["density"], "R_pred": None, "density_pred": None}
                for label, d in docs_by_label.items()
            ],
            "raw": last_err,
        }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return pearson(rx.astype(float), ry.astype(float))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def summarize(batch_results: List[Dict[str, Any]], pool_mean_density: float) -> Dict[str, Any]:
    rank_spearmans = [b["rank_spearman"] for b in batch_results if b["ok"]]
    n_batches_ok = len(rank_spearmans)

    all_per_doc = [d for b in batch_results for d in b["per_doc"]]
    pointwise_ok = [d for d in all_per_doc if d["R_pred"] is not None]

    out: Dict[str, Any] = {
        "n_batches": len(batch_results),
        "n_batches_ok": n_batches_ok,
        "mean_rank_spearman": float(np.mean(rank_spearmans)) if rank_spearmans else float("nan"),
        "n_docs": len(all_per_doc),
        "n_docs_with_estimate": len(pointwise_ok),
    }
    if not pointwise_ok:
        out["pointwise"] = {"error": "no successful per-doc estimates"}
        return out

    M = np.array([d["M"] for d in pointwise_ok], dtype=float)
    R_true = np.array([d["R_true"] for d in pointwise_ok], dtype=float)
    R_pred = np.array([d["R_pred"] for d in pointwise_ok], dtype=float)
    density_true = np.array([d["density_true"] for d in pointwise_ok], dtype=float)
    density_pred = np.array([d["density_pred"] for d in pointwise_ok], dtype=float)
    baseline_R_pred = pool_mean_density * M

    out["pointwise"] = {
        "llm": {
            "pearson_R": pearson(R_true, R_pred), "spearman_R": spearman(R_true, R_pred),
            "r2_R": r2(R_true, R_pred), "mae_R": float(np.mean(np.abs(R_true - R_pred))),
            "pearson_density": pearson(density_true, density_pred),
            "spearman_density": spearman(density_true, density_pred),
            "r2_density": r2(density_true, density_pred),
            "mae_density": float(np.mean(np.abs(density_true - density_pred))),
        },
        "naive_mean_density_baseline": {
            "pool_mean_density": pool_mean_density,
            "pearson_R": pearson(R_true, baseline_R_pred), "r2_R": r2(R_true, baseline_R_pred),
            "mae_R": float(np.mean(np.abs(R_true - baseline_R_pred))),
        },
    }
    return out


async def run_dataset(
    ds_key: str,
    repo_id: str,
    model_id: str,
    base_url: str,
    api_key: str,
    temperature: float,
    n_docs: int,
    batch_size: int,
    concurrency: int,
    retries: int,
    seed: int,
) -> Dict[str, Any]:
    print(f"[{ds_key}] loading train split ({repo_id}) ...", file=sys.stderr)
    pool = load_doc_pool(ds_key, repo_id)
    if not pool:
        return {"dataset": ds_key, "error": "no documents in split"}

    batches = build_percentile_balanced_batches(pool, n_docs, batch_size, seed)
    sample = [doc for batch in batches for doc in batch]
    sample_ids = {d["id"] for d in sample}
    rest_pool = [d for d in pool if d["id"] not in sample_ids] or pool
    pool_mean_density = float(np.mean([d["density"] for d in rest_pool]))

    print(f"[{ds_key}] pool={len(pool)} sampled={len(sample)} batches={len(batches)} "
          f"(batch_size={batch_size}, percentile-balanced) density range=[{min(d['density'] for d in sample):.3f}, "
          f"{max(d['density'] for d in sample):.3f}]", file=sys.stderr)

    llm = ChatOpenAI(model=model_id, temperature=temperature, api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [run_batch(llm, batch, retries, semaphore) for batch in batches]
    t0 = time.time()
    batch_results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"[{ds_key}] done in {elapsed:.1f}s "
          f"({sum(1 for b in batch_results if b['ok'])}/{len(batch_results)} batches ok)", file=sys.stderr)

    summary = summarize(batch_results, pool_mean_density)
    return {"dataset": ds_key, "summary": summary, "batches": batch_results}


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print(f"{'dataset':18s} {'batches':>9s} {'mean rank ρ':>12s} {'n_docs':>7s}"
          f" {'pear(R)':>9s} {'sp(R)':>7s} {'pear(dens)':>11s}  vs baseline R2(R)")
    print("-" * 110)
    for res in results:
        if "error" in res:
            print(f"{res['dataset']:18s}  ERROR: {res['error']}")
            continue
        s = res["summary"]
        pw = s.get("pointwise", {})
        if "error" in pw:
            print(f"{res['dataset']:18s} {s['n_batches_ok']:4d}/{s['n_batches']:<4d}"
                  f" {s['mean_rank_spearman']:12.3f}  (no usable per-doc estimates)")
            continue
        llm_s, base_s = pw["llm"], pw["naive_mean_density_baseline"]
        print(f"{res['dataset']:18s} {s['n_batches_ok']:4d}/{s['n_batches']:<4d}"
              f" {s['mean_rank_spearman']:12.3f} {s['n_docs_with_estimate']:7d}"
              f" {llm_s['pearson_R']:9.3f} {llm_s['spearman_R']:7.3f} {llm_s['pearson_density']:11.3f}"
              f"  {base_s['r2_R']:9.3f}")
    print("=" * 110)


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set.")
    if args.batch_size < 2:
        sys.exit("--batch-size must be >= 2 (ranking needs at least 2 documents to compare).")

    repo_map = load_repo_map()
    ds_keys = DATASET_KEYS if args.datasets == "all" else args.datasets.split(",")
    for k in ds_keys:
        if k not in repo_map:
            sys.exit(f"Unknown dataset key '{k}'. Choose from: {DATASET_KEYS}")

    results = []
    for ds_key in ds_keys:
        res = await run_dataset(
            ds_key=ds_key,
            repo_id=repo_map[ds_key]["repo_id"],
            model_id=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            n_docs=args.n_docs,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            retries=args.retries,
            seed=args.seed,
        )
        results.append(res)

    print_summary_table(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out_path = out_dir / f"density_oracle_{model_slug}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nFull results (incl. per-batch records) written to: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        default_model = yaml.safe_load(f)["model"]["default_model_id"]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model, help="OpenRouter model id to test.")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--datasets", default="all",
                    help=f"Comma-separated subset of {DATASET_KEYS}, or 'all'.")
    p.add_argument("--n-docs", type=int, default=30,
                    help="Total documents sampled per dataset (no document filtering — every doc "
                         "in the split is eligible). Rounded down to a multiple of --batch-size.")
    p.add_argument("--batch-size", type=int, default=4,
                    help="Documents shown together per ranking call, matching the manual subagent "
                         "test exactly. Every batch is percentile-balanced (see "
                         "build_percentile_balanced_batches): one doc from each of `batch_size` "
                         "density bins, so every batch spans low to high density.")
    p.add_argument("--concurrency", type=int, default=10, help="Concurrent batch calls.")
    p.add_argument("--retries", type=int, default=2, help="Retries per batch on parse failure.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="logs/density_oracle")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
