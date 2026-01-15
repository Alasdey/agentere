# main.py
import asyncio
import json
import yaml
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools import get_enabled_tools
from utils.resample import aggregate_run_triples
from utils.metrics import compute_ere_metrics
from langchain_core.messages import SystemMessage, HumanMessage

# --- Load Config ---
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

active_ds_key = CONFIG["active_dataset"]
ds_cfg = CONFIG["datasets"][active_ds_key]
prompt_cfg = CONFIG["prompts"][ds_cfg["prompt"]]
exp_cfg = CONFIG.get("experiment", {})

# --- Parsing Helper with Strict Validation ---
def parse_llm_json(ai_msg) -> List[Tuple[str, str, str]]:
    content = ai_msg.content
    if not content:
        raise ValueError("Empty response from LLM")
        
    try:
        # Locate JSON array
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in output")
            
        json_str = content[start:end]
        data = json.loads(json_str)
        
        # Transform and validate shape
        triples = []
        for obj in data:
            if "pair" in obj and "label" in obj:
                parts = obj["pair"].split(",")
                if len(parts) == 2:
                    triples.append((parts[0].strip(), obj["label"].strip(), parts[1].strip()))
        return triples
    except Exception as e:
        # Re-raise to trigger the retry loop
        raise ValueError(f"JSON Parsing failed: {str(e)}")

# --- Robust Single Sample with Retries ---
async def run_single_sample_no_retry(doc, ainvoke_func, thread_id) -> List[Tuple[str, str, str]]:
    """
    Directly calls the graph and parser. 
    Does NOT catch exceptions (JSON or API), allowing them to crash for debugging.
    """
    sys_msg = SystemMessage(prompt_cfg["system"])
    user_msg = HumanMessage(prompt_cfg["user_template"].format(
        doc_text=doc["doc_text"],
        event_list="Extracted from tags"
    ))
    config = {"configurable": {"thread_id": thread_id}}
    
    state = await ainvoke_func([sys_msg, user_msg], config=config)
    # If parse_llm_json raises ValueError, it is NOT caught here
    return parse_llm_json(state["messages"][-1])

# --- Resampling Execution ---
async def process_document_resampled(doc, ainvoke_func, semaphore):
    """
    Switchboard: Robust resampling vs. Debug single-run.
    """
    resamp_cfg = exp_cfg.get("resampling", {})
    is_resampling_enabled = resamp_cfg.get("enabled", False)

    async with semaphore:
        if not is_resampling_enabled:
            # DEBUG MODE: Single run, no retries, crash on error
            final_preds = await run_single_sample_no_retry(
                doc, ainvoke_func, f"{doc['id']}_debug"
            )
            return doc["gold_triples"], final_preds
        
        # PRODUCTION MODE: Multiple runs with individual retry logic
        n_runs = resamp_cfg.get("n_runs", 1)
        sampling_tasks = [
            run_single_sample_with_retry(
                doc, 
                ainvoke_func, 
                f"{doc['id']}_run{i}_{uuid.uuid4().hex[:4]}"
            )
            for i in range(n_runs)
        ]
        
        all_run_outputs = await asyncio.gather(*sampling_tasks)
        
        # Aggregation
        final_preds = aggregate_run_triples(
            all_run_outputs, 
            tie_breaking=resamp_cfg.get("tie_breaking", "norel")
        )
        
        return doc["gold_triples"], final_preds

# --- Main Entry ---
async def main():
    # 1. Dynamic Tool Loading
    enabled_tool_names = exp_cfg.get("tools", [])
    tools = get_enabled_tools(enabled_tool_names)

    # 2. Build Modular Graph
    graph, invoke, ainvoke = build_chat_graph(
        model_id=CONFIG["model"]["default_model_id"],
        temperature=CONFIG["model"]["temperature"],
        enable_tools=exp_cfg.get("enable_tools", True),
        tools=tools
    )

    # 3. Load Dataset
    dataset = list(load_hf_dataset_parsed(
        repo_id=ds_cfg["repo_id"],
        split=ds_cfg["split"],
        text_field=ds_cfg["text_field"],
        ann_field=ds_cfg["ann_field"],
        max_examples=ds_cfg.get("max_examples", 0)
    ))

    # 4. Run loop
    concurrency = exp_cfg.get("concurrency", 5)
    sem = asyncio.Semaphore(concurrency)
    tasks = [process_document_resampled(doc, ainvoke, sem) for doc in dataset]
    
    print(f"Engine started. Dataset: {active_ds_key} | Resampling: {exp_cfg.get('resampling', {}).get('n_runs', 1)}x | Retries: {exp_cfg.get('retries', 3)}")
    results = await asyncio.gather(*tasks)

    # 5. Global Metrics
    all_gold, all_pred = [], []
    for g, p in results:
        all_gold.extend(g)
        all_pred.extend(p)

    final_metrics = compute_ere_metrics(all_gold, all_pred)
    
    print("\n" + "="*40)
    print(f"PERFORMANCE SUMMARY: {active_ds_key}")
    print(f"Total Docs: {len(dataset)}")
    print(f"F1-Score:   {final_metrics['f1']:.4f}")
    print(f"Precision:  {final_metrics['precision']:.4f}")
    print(f"Recall:     {final_metrics['recall']:.4f}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())