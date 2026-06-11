#!/usr/bin/env python3
"""
Sample incorrect pair predictions, then use an LLM to extract the relevant
reasoning passage from the model's CoT for each wrong pair.

Usage:
    # Single run
    uv run python scripts/analysis/pair_cot_errors_llm.py logs/allatonce/run_XYZ.json --n-pairs 20

    # Latest run only
    uv run python scripts/analysis/pair_cot_errors_llm.py --latest logs/allatonce/ --n-pairs 20

    # Custom model / base-url
    uv run python scripts/analysis/pair_cot_errors_llm.py --latest logs/allatonce/ --n-pairs 20 \
        --model anthropic/claude-haiku-4-5 --base-url https://openrouter.ai/api/v1

    # Write to file
    uv run python scripts/analysis/pair_cot_errors_llm.py --latest logs/allatonce/ --n-pairs 20 --out errors.json
"""
import argparse
import glob
import json
import os
import random
import sys
import yaml
from collections import defaultdict
from datasets import load_dataset
from openai import OpenAI


# ── Config / trace loading ────────────────────────────────────────────────────

def load_config(json_path: str) -> dict:
    config_path = json_path.replace(".json", ".config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_traces(json_path: str) -> list[list[dict]]:
    traces_path = json_path.replace(".json", ".traces.jsonl")
    if not os.path.exists(traces_path):
        return []
    with open(traces_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_ds_key(config: dict) -> tuple[str, str, str]:
    active = config["active_dataset"]
    ds_cfg = config["datasets"][active]
    return ds_cfg["repo_id"], ds_cfg["split"], ds_cfg["text_field"]


# ── HuggingFace doc index ─────────────────────────────────────────────────────

def build_mentions_map(row: dict) -> dict[str, str]:
    tokens = row["tokens"]
    spans = row["spans"]
    mentions = row["mentions"]
    result = {}
    for j, mention_id in enumerate(mentions):
        parts = [tokens[idx] for idx in spans[j] if 0 <= idx < len(tokens)]
        if parts:
            result[str(mention_id)] = " ".join(parts)
    return result


def load_doc_info(repo_id: str, split: str, text_field: str) -> dict[str, dict]:
    """Return {doc_id: {"text": tagged_text, "mentions_map": {id: text}}}."""
    ds = load_dataset(repo_id, split=split)
    docs = {}
    for i in range(len(ds)):
        row = ds[i]
        doc_id = str(row["id"])
        docs[doc_id] = {
            "text": row[text_field],
            "mentions_map": build_mentions_map(row),
        }
    return docs


def format_pair(pair_str: str, mentions_map: dict[str, str]) -> str:
    """Format 'A,B' as 'A text_a, B text_b' when mention texts are available."""
    src_id, tgt_id = [x.strip() for x in pair_str.split(",", 1)]
    src_text = mentions_map.get(src_id, "")
    tgt_text = mentions_map.get(tgt_id, "")
    if src_text and tgt_text:
        return f"{src_id} {src_text}, {tgt_id} {tgt_text}"
    return pair_str


# ── Genuine-extraction detection ──────────────────────────────────────────────

def get_genuine_extraction(trace: list[dict], user_template: str) -> tuple[str, str] | None:
    """
    Walk a trace and return (human_content, ai_content) for the LAST (human, ai)
    pair where the human message contains the user_template prefix.

    This filters out:
    - Fewshot (human, ai) pairs that precede the genuine call — they also match
      user_template, but only the last one is the genuine extraction.
    - Synth CoT generation messages ("The correct label assignments …",
      "Rewrite your response …") — those don't match user_template.
    """
    template_prefix = user_template[:60]
    pairs: list[tuple[str, str]] = []

    i = 0
    while i < len(trace):
        msg = trace[i]
        kwargs = msg.get("kwargs", {})
        role = kwargs.get("type", kwargs.get("role", ""))
        content = kwargs.get("content", "")

        if role == "human" and isinstance(content, str) and template_prefix in content:
            if i + 1 < len(trace):
                nxt = trace[i + 1]
                nxt_kwargs = nxt.get("kwargs", {})
                nxt_role = nxt_kwargs.get("type", nxt_kwargs.get("role", ""))
                if nxt_role == "ai":
                    pairs.append((content, nxt_kwargs.get("content", "")))
                    i += 2
                    continue
        i += 1

    return pairs[-1] if pairs else None


# ── Trace ↔ doc matching via text substring ───────────────────────────────────

def build_trace_cot_map(
    traces: list[list[dict]],
    user_template: str,
    doc_info: dict[str, dict],
) -> dict[str, list[str]]:
    """
    Build {doc_id: [ai_cot, …]} by matching the HuggingFace doc text as a
    substring inside each trace's genuine extraction human message.
    """
    result: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        gen = get_genuine_extraction(trace, user_template)
        if gen is None:
            continue
        human_content, ai_content = gen
        for doc_id, info in doc_info.items():
            text = info["text"]
            if text and text in human_content:
                result[doc_id].append(ai_content)
                break
    return result


# ── LLM-based CoT reasoning extraction ───────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are an assistant that locates reasoning passages in a model's chain-of-thought (CoT).
Given a document text, a pair of event mentions, and the full CoT produced by the model,
return the verbatim passage(s) from the CoT that explain why the model assigned the
label it did to that specific pair. Quote directly — do not paraphrase or summarise.
If no passage explicitly addresses the pair, return the most relevant context you can find.
The most relevant context maybe that the model stoped enumerating after a blanket statement or discarded one of the mention before even examining the pair.
Also do not write anything else than a direct quote from the text. No 'The verbatim passage is' no nothing.
Do not give a passage corresponding to another pair. In this case no answer is better than the wrong answer.
If no passage at all is relevant give the passage that decided which mentions to still consider.
And if that is still not relevant simply say 'Undiscussed' and nothing else"""

EXTRACTION_USER = """\
Document text:
{text}

Pair: {pair}
Gold label: {gold}
Predicted label: {pred}

Full CoT:
{cot}

Extract the verbatim passage(s) from the CoT that explain the model's reasoning for this pair."""


def llm_extract_cot(
    client: OpenAI,
    model: str,
    text: str,
    pair: str,
    gold: str,
    pred: str,
    cot: str,
) -> str:
    prompt = EXTRACTION_USER.format(
        text=text,
        pair=pair,
        gold=gold,
        pred=pred,
        cot=cot,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(
    log_paths: list[str],
    n_pairs: int,
    client: OpenAI,
    model: str,
    out_path: str | None = None,
    seed: int = 42,
) -> dict:
    print(f"Loading {len(log_paths)} log file(s)…")

    all_wrong: list[dict] = []

    for log_path in log_paths:
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)

        config = load_config(log_path)
        user_template = config["prompt"]["user_template"]
        traces = load_traces(log_path)

        ppp: list[dict] = data["results"]["per_pair_predictions"]

        repo_id, split, text_field = get_ds_key(config)
        print(f"  Fetching texts from {repo_id} ({split})…")
        doc_info = load_doc_info(repo_id, split, text_field)

        doc_id_to_cots: dict[str, list[str]] = (
            build_trace_cot_map(traces, user_template, doc_info) if traces else {}
        )

        unique_doc_ids = {p["id"] for p in ppp}
        n_matched = sum(1 for did in unique_doc_ids if did in doc_id_to_cots)
        n_total = len(unique_doc_ids)
        if traces:
            print(f"  {os.path.basename(log_path)}: {n_matched}/{n_total} docs matched to traces")

        for p in ppp:
            gold = p["gold"]
            pred = p["pred"]
            vote_counts: dict = p.get("vote_counts") or {}
            majority = max(vote_counts, key=vote_counts.__getitem__) if vote_counts else pred

            if majority == gold:
                continue

            doc_id = p["id"]
            info = doc_info.get(doc_id, {})
            mentions_map = info.get("mentions_map", {})
            cots = doc_id_to_cots.get(doc_id, [])

            all_wrong.append(
                {
                    "log": os.path.basename(log_path),
                    "doc_idx": p["doc_idx"],
                    "doc_id": doc_id,
                    "text": info.get("text", ""),
                    "pair": format_pair(p["pair"], mentions_map),
                    "pair_raw": p["pair"],
                    "gold": gold,
                    "pred": majority,
                    "vote_counts": vote_counts or None,
                    "cot": cots[0] if cots else None,
                }
            )

    print(f"  Total wrong pairs: {len(all_wrong)}")

    rng = random.Random(seed)
    sample = rng.sample(all_wrong, min(n_pairs, len(all_wrong)))

    print(f"  Calling LLM ({model}) for {len(sample)} pairs…")
    for i, entry in enumerate(sample):
        cot = entry.pop("cot")
        pair_raw = entry.pop("pair_raw")
        if cot:
            entry["cot_reasoning"] = llm_extract_cot(
                client, model,
                text=entry["text"],
                pair=entry["pair"],
                gold=entry["gold"],
                pred=entry["pred"],
                cot=cot,
            )
        else:
            entry["cot_reasoning"] = None
        print(f"    {i + 1}/{len(sample)}", end="\r", flush=True)
    print()

    result = {
        "logs": [os.path.basename(p) for p in log_paths],
        "model": model,
        "total_wrong": len(all_wrong),
        "n_pairs_sampled": len(sample),
        "sample": sample,
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Written to: {out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def get_active_dataset(log_path: str) -> str:
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)["config"]["active_dataset"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample incorrect predictions with LLM-extracted CoT reasoning"
    )
    parser.add_argument("logs", nargs="*", help="Log JSON file(s) or directory")
    parser.add_argument("--n-pairs", type=int, default=20, help="Number of wrong pairs to sample")
    parser.add_argument("--latest", type=int, metavar="N", help="Use only the N most recent logs")
    parser.add_argument("--dataset", help="Filter logs to this active_dataset key")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out", help="Write output JSON to this path")
    parser.add_argument(
        "--model",
        # default="anthropic/claude-haiku-4-5",
        default="qwen/qwen3-30b-a3b-instruct-2507",
        help="Model for CoT extraction",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenAI-compatible base URL (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="API key (default: $OPENROUTER_API_KEY)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: no API key — set $OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    paths: list[str] = []
    for p in args.logs:
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, "run_*.json")))
        elif "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)

    paths = [
        p for p in paths
        if not any(x in p for x in (".traces.", ".config.", ".diff."))
    ]

    if not paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(set(paths), key=os.path.getmtime)

    if args.dataset:
        paths = [p for p in paths if get_active_dataset(p) == args.dataset]
        if not paths:
            print(f"No logs found for dataset '{args.dataset}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Filtered to {len(paths)} log(s) for dataset '{args.dataset}'.")

    if args.latest:
        paths = paths[-args.latest:]

    try:
        analyze(
            paths,
            n_pairs=args.n_pairs,
            client=client,
            model=args.model,
            out_path=args.out,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
