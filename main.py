# main.py
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage

from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools import get_enabled_tools
from tools.encoder import CURRENT_DOC_ID
from tools.few_shot import CURRENT_DOC_TEXT, CURRENT_DOC_MENTIONS
from utils.config import load_config
from utils.formatting import format_pair_lines
from utils.logger import log_experiment
from utils.metrics import compute_ere_metrics
from utils.reporting import generate_run_report
from utils.resample import aggregate_run_triples
import utils.trace_dump as trace_dump


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_llm_json(message: BaseMessage) -> List[Tuple[str, str, str]]:
    """Parses the LLM response message into a list of (src, label, tgt) triples."""
    content = message.content
    try:
        # Regex to find JSON array in case of conversational filler
        match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in output")
        
        data = json.loads(match.group(0))
        triples = []
        for item in data:
            pair = item.get("pair", "")
            if "," in pair:
                src, tgt = [part.strip() for part in pair.split(",", 1)]
                triples.append((src, item.get("label", "Unknown"), tgt))
        return triples
    except Exception as e:
        raise ValueError(f"JSON Parsing failed: {str(e)}")

# =============================================================================
# CORE EXECUTION LOGIC
# =============================================================================

async def run_single_inference(graph_ainvoke, system_prompt, user_prompt, few_shot_str: str = "") -> Dict[str, Any]:
    """Executes a single pass and returns parsed triples + raw content."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    if few_shot_str:
        call_id = uuid.uuid4().hex[:8]
        messages.extend([
            AIMessage(content="", tool_calls=[{"id": call_id, "name": "few_shot_examples", "args": {}}]),
            ToolMessage(content=few_shot_str, tool_call_id=call_id),
        ])
    state = await graph_ainvoke(messages)
    
    trace_dump.trace_dump(state)

    # The last message contains the final answer
    raw_content = state["messages"][-1].content
    
    # helper for internal logic
    parsed_triples = parse_llm_json(state["messages"][-1]) 
    
    return {
        "triples": parsed_triples,
        "raw_response": raw_content
    }

async def run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, max_retries: int, few_shot_str: str = ""):
    """Retries inference if parsing fails, returns dict with raw+parsed data."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await run_single_inference(graph_ainvoke, system_prompt, user_prompt, few_shot_str)
        except ValueError as e:
            print(f'Failed inference {attempt}: {e}')
            last_error = e
            await asyncio.sleep(1)
            
    raise ValueError(f"Max retries reached. Last error: {last_error}")

async def process_document_resampled(doc, config, graph_ainvoke):
    """
    Runs inference (optionally resampled) for a single document.
    Returns inputs, outputs, stats, metrics, AND full context trace.
    """
    n_runs = config["experiment"]["resampling"]["n_runs"] if config["experiment"]["resampling"]["enabled"] else 1
    retries = config["experiment"].get("retries", 3)
    
    prompt_cfg = config["prompt"]
    system_prompt = prompt_cfg["system"]
    active_ds = config["active_dataset"]

    pair_lines = format_pair_lines(doc, active_ds)

    user_prompt = prompt_cfg["user_template"].format(
        doc_text=doc["doc_text"],
        pair_lines=pair_lines,
        doc_id=doc["id"],
    )

    # ── Set document context (needed by tools and similarity-based few-shot) ──
    ctx_token = CURRENT_DOC_ID.set(doc["id"])
    ctx_token_text = CURRENT_DOC_TEXT.set(doc["doc_text"])
    ctx_token_mentions = CURRENT_DOC_MENTIONS.set(frozenset(doc.get("mentions_map", {}).values()))

    # ── Few-shot systematic injection ────────────────────────────────────────
    fs_cfg = config.get("few_shot", {})
    few_shot_str = ""
    if fs_cfg.get("enabled") and fs_cfg.get("systematic", True):
        from tools.few_shot import few_shot_examples
        few_shot_str = await few_shot_examples.ainvoke({})

    # 1. Run inference N times
    sampling_tasks = [
        run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, retries, few_shot_str)
        for _ in range(n_runs)
    ]
    
    try:
        # runs_results will be a list of dicts: [{'triples': [...], 'raw_response': "..."}]
        runs_results = await asyncio.gather(*sampling_tasks)
        
        # Extract just the triples for aggregation logic
        triples_only_lists = [r["triples"] for r in runs_results]
        
        # Aggregate using voting logic
        pair_stats = {}
        if len(triples_only_lists) > 1:
            final_preds, pair_stats = aggregate_run_triples(
                triples_only_lists, 
                tie_breaking=config["experiment"]["resampling"].get("tie_breaking", "norel")
            )
        else:
            final_preds = triples_only_lists[0]
            # Dummy stats for single run
            pair_stats = {} 
            for src, lbl, tgt in final_preds:
                pair_stats[f"{src},{tgt}"] = {"vote_counts": {lbl: 1}}

        # 2. Metrics
        metrics = compute_ere_metrics(doc["gold_triples"], final_preds)
        
        # 3. Build Context Object for Auditing
        # We collect all raw responses from the N runs
        context_trace = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_responses": [r["raw_response"] for r in runs_results]
        }
        
        return {
            "id": doc["id"],
            "doc_idx": doc["doc_idx"],
            "lang": doc.get("lang", "unknown"),
            "gold_triples": doc["gold_triples"],
            "pred_triples": final_preds,
            "pair_stats": pair_stats, 
            "metrics": metrics,
            "context": context_trace  # <--- NEW FIELD containing raw traces
        }
    except Exception as e:
        print(f"Document {doc['id']} failed after all retries: {e}")
        return None
    finally:
        CURRENT_DOC_ID.reset(ctx_token)
        CURRENT_DOC_TEXT.reset(ctx_token_text)
        CURRENT_DOC_MENTIONS.reset(ctx_token_mentions)


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

async def main():
    # 1. Load Config
    config = load_config()

    os.environ["LANGCHAIN_TRACING_V2"] = config["experiment"]["tracing"]
    os.environ["LANGCHAIN_PROJECT"] = config["experiment"]["tracing_name"]

    # Dataset
    active_ds_key = config["active_dataset"]
    ds_config = config["datasets"][active_ds_key]

    # Labels types
    active_labels = ds_config.get("labels")
    print(f"Active Labels for {ds_config['name']}: {active_labels}")

    # 2. Build Graph
    tools = get_enabled_tools(config["experiment"].get("tools", []))
    _, graph_ainvoke = build_chat_graph(
        model_id=config["model"]["default_model_id"],
        temperature=config["model"]["temperature"],
        base_url=config["model"]["base_url"],
        tools=tools,
        enable_tools=config["experiment"]["enable_tools"]
    )

    print(f"Engine started. Dataset: {ds_config['name']} | Samples: {ds_config['max_examples']} | Retries: {config['experiment'].get('retries', 0)}")

    # 3. Load Data
    dataset_iter = load_hf_dataset_parsed(
        repo_id=ds_config["repo_id"],
        split=ds_config["split"],
        text_field=ds_config["text_field"],
        ann_field=ds_config["ann_field"],
        max_examples=ds_config.get("max_examples", 0),
        valid_labels=set(active_labels),
    )

    # 4. Execute Concurrently
    semaphore = asyncio.Semaphore(config["experiment"].get("concurrency", 10))
    completed = 0

    async def sem_task(doc, total):
        nonlocal completed
        async with semaphore:
            result = await process_document_resampled(doc, config, graph_ainvoke)
        completed += 1
        print(f"[{completed}/{total}] Processed doc {doc['id']}", flush=True)
        return result

    docs = list(dataset_iter)
    total = len(docs)
    tasks = [sem_task(doc, total) for doc in docs]
    results = await asyncio.gather(*tasks)

    # 5. Generate Report (Using new module)
    print("Aggregating results and computing metrics...")
    
    final_report = generate_run_report(
        results=results,
        total_processed_count=total,
        config=config,
        valid_labels=set(active_labels),
        # Optional: Override labels if needed, otherwise uses default ERE set
        # valid_labels=["CauseEffect", "EffectCause", "CAUSE", "PRECONDITION", "NoRel"]
    )

    print(f"Eval completed. Metric F1: {final_report['micro_f1']:.4f}")

    # 6. Log to Disk
    outfile = log_experiment(
        logdir="logs/allatonce",
        config=config,
        cli_args=sys.argv,
        results=final_report,
        filename_prefix="run",
    )
    print(f"Results logged to: {outfile}")
    keys = ["per_label", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "total_pairs", "binary"]
    for key in keys:
        print(key, ":", final_report[key])

if __name__ == "__main__":
    asyncio.run(main())