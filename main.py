# main.py
import asyncio
import json
import yaml
import uuid
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os

from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools import get_enabled_tools
from utils.resample import aggregate_run_triples
from utils.metrics import compute_ere_metrics
from langchain_core.messages import SystemMessage, HumanMessage

# --- Load Config ---
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

os.environ["LANGCHAIN_TRACING_V2"] = CONFIG["experiment"]["tracing"] 
os.environ["LANGCHAIN_PROJECT"] = CONFIG["experiment"]["tracing_name"]

active_ds_key = CONFIG["active_dataset"]
ds_cfg = CONFIG["datasets"][active_ds_key]
prompt_cfg = CONFIG["prompts"][ds_cfg["prompt"]]
exp_cfg = CONFIG.get("experiment", {})

import asyncio
import yaml
import json
import re
import os
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools import get_enabled_tools
from utils.metrics import compute_ere_metrics
from utils.resample import aggregate_run_triples

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

async def run_single_inference(graph_ainvoke, system_prompt, user_prompt) -> List[Tuple[str, str, str]]:
    """Executes a single pass through the LangGraph and parses JSON."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    state = await graph_ainvoke(messages)
    # The last message in state contains the final answer after tool loops
    return parse_llm_json(state["messages"][-1])

async def run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, max_retries: int):
    """Retries the inference if parsing fails."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await run_single_inference(graph_ainvoke, system_prompt, user_prompt)
        except ValueError as e:
            last_exception = e
            print(f"  [Attempt {attempt+1}/{max_retries+1}] Parsing failed: {e}")
            await asyncio.sleep(1) 
    raise last_exception

async def process_document_resampled(doc, config, ds_config, graph_ainvoke):
    """Handles resampling (N votes per doc) and retries per vote."""
    n_runs = config["experiment"]["resampling"]["n_runs"] if config["experiment"]["resampling"]["enabled"] else 1
    retries = config["experiment"].get("retries", 3)
    
    prompt_cfg = config["prompts"][ds_config["prompt"]]
    system_prompt = prompt_cfg["system"]
    
    # Simple event listing placeholder for cleaner prompts
    user_prompt = prompt_cfg["user_template"].format(
        doc_text=doc["doc_text"],
        pair_lines="" # Used by MECI-style prompts
    )

    sampling_tasks = [
        run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, retries)
        for _ in range(n_runs)
    ]
    
    try:
        # Run all samples for this document in parallel
        runs_outputs = await asyncio.gather(*sampling_tasks)
        
        # Aggregate using voting logic from utils/resample.py
        if len(runs_outputs) > 1:
            final_preds = aggregate_run_triples(
                runs_outputs, 
                tie_breaking=config["experiment"]["resampling"].get("tie_breaking", "norel")
            )
        else:
            final_preds = runs_outputs[0]
            
        metrics = compute_ere_metrics(doc["gold_triples"], final_preds)
        return {"id": doc["id"], "metrics": metrics, "preds": final_preds}
    
    except Exception as e:
        print(f"Document {doc['id']} failed after all retries: {e}")
        return None

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

async def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    active_ds_key = config["active_dataset"]
    ds_config = config["datasets"][active_ds_key]
    
    # Initialize LangGraph with defined tools
    tools = get_enabled_tools(config["experiment"].get("tools", []))
    _, _, graph_ainvoke = build_chat_graph(
        model_id=config["model"]["default_model_id"],
        temperature=config["model"]["temperature"],
        base_url=config["model"]["base_url"],
        tools=tools,
        enable_tools=config["experiment"]["enable_tools"]
    )

    print(f"Engine started. Dataset: {ds_config['name']} | Retries: {config['experiment'].get('retries', 0)}")

    dataset_iter = load_hf_dataset_parsed(
        repo_id=ds_config["repo_id"],
        split=ds_config["split"],
        text_field=ds_config["text_field"],
        ann_field=ds_config["ann_field"],
        max_examples=ds_config.get("max_examples", 0)
    )

    # Process documents concurrently based on config
    semaphore = asyncio.Semaphore(config["experiment"].get("concurrency", 10))
    
    async def sem_task(doc):
        async with semaphore:
            return await process_document_resampled(doc, config, ds_config, graph_ainvoke)

    tasks = [sem_task(doc) for doc in dataset_iter]
    results = await asyncio.gather(*tasks)
    
    # filter None and report aggregate
    valid_results = [r for r in results if r]
    avg_f1 = sum(r["metrics"]["f1"] for r in valid_results) / len(valid_results) if valid_results else 0
    print(f"Eval completed. Avg F1: {avg_f1:.4f}")

if __name__ == "__main__":
    asyncio.run(main())