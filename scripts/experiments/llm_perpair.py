# main_pairwise.py
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from itertools import combinations

from langchain_core.messages import SystemMessage, HumanMessage
from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools import get_enabled_tools
from utils.config import load_config
from utils.metrics import compute_ere_metrics
from utils.logger import log_experiment
from utils.reporting import generate_run_report
from tools.encoder import CURRENT_DOC_ID

# --- Load Config ---
CONFIG = load_config()

os.environ["LANGCHAIN_TRACING_V2"] = CONFIG["experiment"]["tracing"]
os.environ["LANGCHAIN_PROJECT"] = CONFIG["experiment"]["tracing_name"]

active_ds_key = CONFIG["active_dataset"]
ds_cfg = CONFIG["datasets"][active_ds_key]
prompt_cfg = CONFIG["prompt"]

# =============================================================================
# CORE PAIRWISE LOGIC
# =============================================================================

async def classify_single_pair(graph_ainvoke, system_prompt, user_template, doc_text, pair_str, mentions_map):
    """
    Runs inference for a single pair.
    """
    # Parse pair IDs
    parts = pair_str.split(",")
    if len(parts) != 2:
        return {"pair": pair_str, "label": "NoRel"}
    
    src, tgt = parts[0].strip(), parts[1].strip()
    
    # Format the single pair line as expected by the prompt
    # Using the style found in main.py for MECI/MAVEN
    src_quote = mentions_map.get(src, "")
    tgt_quote = mentions_map.get(tgt, "")
    pair_line = f'{src} ("{src_quote}"), {tgt} ("{tgt_quote}")'
    
    # Generate User Prompt
    user_prompt = user_template.format(
        doc_text=doc_text,
        pair_lines=pair_line, # Inject ONLY the current pair
        doc_id="" # Not strictly needed for prompt but passed if template uses it
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        state = await graph_ainvoke(messages)
        content = state["messages"][-1].content
        
        # Parse JSON
        # The model is asked to return a JSON array.
        # We extract the JSON object inside the array.
        import re
        # match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
        match = re.search(r"\s*\{.*\}\s*", content, re.DOTALL)
        # print("match inplace:", match, end="\r")
        if match:
            group = match.group(0)
            data = json.loads(group)
            if isinstance(data, dict) and len(data) > 0:
                item = data
                print("item:", item, " "*10, end="\r")
                return item
        
        # Fallback if parsing fails or empty list
        return {"pair": pair_str, "label": "NoRel"}
        
    except Exception as e:
        print(f"Error classifying pair {pair_str}: {e}")
        print("-"*20)
        print(f"Messages", messages)
        print("-"*20)
        print("match:", match)
        print("-"*20)
        print("group:", group)
        print("-"*20)
        print("data:", data)
        return {"pair": pair_str, "label": "NoRel"}

async def process_document_pairwise(doc, config, ds_config, graph_ainvoke):
    """
    Generates all candidate pairs for a document and classifies them one by one.
    """
    prompt_cfg = config["prompt"]
    system_prompt = prompt_cfg["system"]
    user_template = prompt_cfg["user_template"]
    
    mentions_map = doc.get("mentions_map", {})
    mention_ids = list(mentions_map.keys())
    
    if not mention_ids:
        # If no mentions, return empty
        return {
            "id": doc["id"],
            "doc_idx": doc["doc_idx"],
            "lang": doc.get("lang", "unknown"),
            "gold_triples": doc["gold_triples"],
            "pred_triples": [],
            "metrics": compute_ere_metrics(doc["gold_triples"], []),
            "context": {}
        }

    # 1. Generate all candidate pairs (i < j) for Directed Graph, or all permutations
    # Most datasets require checking directionality (A->B vs B->A).
    # We will generate all ordered pairs for safety in causality tasks.
    candidate_pairs = []
    for i in range(len(mention_ids)):
        for j in range(len(mention_ids)):
            if i != j:
                src = mention_ids[i]
                tgt = mention_ids[j]
                candidate_pairs.append(f"{src},{tgt}")
    
    # Note: For large documents, this is O(N^2) API calls. 
    # Minimal implementation: Just run them all.
    
    # 2. Set document context for tools (e.g. encoder)
    ctx_token = CURRENT_DOC_ID.set(doc["id"])
    
    # 3. Create tasks
    tasks = [
        classify_single_pair(graph_ainvoke, system_prompt, user_template, doc["doc_text"], p, mentions_map)
        for p in candidate_pairs
    ]
    
    try:
        # Run all pair classifications for this document concurrently
        # WARNING: This can hit rate limits very fast for docs with >10 mentions.
        # Using semaphore inside classify_single_pair or batching is recommended for production.
        raw_results = await asyncio.gather(*tasks)
        
        # 4. Filter NoRel and aggregate
        final_preds = []
        for res in raw_results:
            lbl = res.get("label", "NoRel")
            if lbl not in ["NoRel", "Unknown", ""]:
                p_str = res.get("pair", "")
                if "," in p_str:
                    s, t = [x.strip() for x in p_str.split(",", 1)]
                    final_preds.append((s, lbl, t))

        metrics = compute_ere_metrics(doc["gold_triples"], final_preds)
        print(final_preds)
        print(metrics)

        return {
            "id": doc["id"],
            "doc_idx": doc["doc_idx"],
            "lang": doc.get("lang", "unknown"),
            "gold_triples": doc["gold_triples"],
            "pred_triples": final_preds,
            "pair_stats": {}, # No voting in single pass
            "metrics": metrics,
            "context": {
                "n_pairs_processed": len(candidate_pairs)
            }
        }
        
    except Exception as e:
        print(f"Document {doc['id']} failed: {e}")
        return None
    finally:
        CURRENT_DOC_ID.reset(ctx_token)

# =============================================================================
# MAIN
# =============================================================================

async def main():
    # 1. Setup
    tools = get_enabled_tools(CONFIG["experiment"].get("tools", []))
    _, graph_ainvoke = build_chat_graph(
        model_id=CONFIG["model"]["default_model_id"],
        temperature=CONFIG["model"]["temperature"],
        base_url=CONFIG["model"]["base_url"],
        tools=tools,
        enable_tools=CONFIG["experiment"]["enable_tools"]
    )

    print(f"Pairwise Extraction Mode. Dataset: {ds_cfg['name']} | Samples: {ds_cfg.get('max_examples', 'All')}")

    # 2. Load Data
    dataset_iter = load_hf_dataset_parsed(
        repo_id=ds_cfg["repo_id"],
        split=ds_cfg["split"],
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        max_examples=ds_cfg.get("max_examples", 0),
        valid_labels=set(ds_cfg.get("labels", [])),
    )

    # 3. Execute
    # To prevent overwhelming the API, we process documents sequentially, 
    # but pairs within a document concurrently.
    # For high scale, use an external semaphore.
    
    semaphore = asyncio.Semaphore(CONFIG["experiment"].get("concurrency", 5))
    
    async def sem_task(doc):
        async with semaphore:
                return await process_document_pairwise(doc, CONFIG, ds_cfg, graph_ainvoke)

    tasks = [sem_task(doc) for doc in dataset_iter]
    results = await asyncio.gather(*tasks)

    # 4. Report
    print("Generating report...")
    active_labels = ds_cfg.get("labels")
    
    final_report = generate_run_report(
        results=results,
        total_processed_count=len(tasks),
        config=CONFIG,
        valid_labels=set(active_labels) if active_labels else None
    )

    print(f"Eval completed. F1: {final_report['micro_f1']:.4f}")
    
    # 5. Log
    outfile = log_experiment(
        logdir="logs/perpair",
        config=CONFIG,
        cli_args=sys.argv,
        results=final_report,
        filename_prefix="run_pairwise",
    )
    print(f"Logged to: {outfile}")

if __name__ == "__main__":
    asyncio.run(main())