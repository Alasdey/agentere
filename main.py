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
from tools.few_shot import CURRENT_DOC_TEXT, CURRENT_DOC_MENTIONS, CURRENT_DOC_FOLD, preload as few_shot_preload, get_few_shot_message_pairs
from tools.reprompt import CURRENT_USER_PROMPT
from utils.config import load_config
from utils.formatting import format_pair_lines
from utils.logger import log_experiment, make_run_stem, capture_git_state
from utils.metrics import compute_ere_metrics
from utils.mlflow_tracker import log_run as mlflow_log_run
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

async def run_single_inference(graph_ainvoke, system_prompt, user_prompt, few_shot_pairs: list = None, reprompt_str: str = "") -> Dict[str, Any]:
    """Executes a single pass and returns parsed triples + raw content."""
    messages = [SystemMessage(content=system_prompt)]
    for human_content, ai_content in (few_shot_pairs or []):
        messages.append(HumanMessage(content=human_content))
        messages.append(AIMessage(content=ai_content))
    messages.append(HumanMessage(content=user_prompt))
    if reprompt_str:
        call_id = uuid.uuid4().hex[:8]
        messages.extend([
            AIMessage(content="", tool_calls=[{"id": call_id, "name": "reprompt", "args": {}}]),
            ToolMessage(content=reprompt_str, tool_call_id=call_id),
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

async def run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, max_retries: int, few_shot_pairs: list = None, reprompt_str: str = "", doc_id: str = "?", timeout: int = 3600):
    """Retries inference if parsing fails or times out, returns dict with raw+parsed data."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(
                run_single_inference(graph_ainvoke, system_prompt, user_prompt, few_shot_pairs, reprompt_str),
                timeout=timeout,
            )
        except (ValueError, asyncio.TimeoutError) as e:
            print(f'[{doc_id}] Failed inference attempt {attempt}: {e}')
            last_error = e
            await asyncio.sleep(1)

    raise ValueError(f"[{doc_id}] Max retries reached. Last error: {last_error}")

async def process_document_resampled(doc, config, graph_ainvoke):
    """
    Runs inference (optionally resampled) for a single document.
    Returns inputs, outputs, stats, metrics, AND full context trace.
    """
    n_runs = config["experiment"]["resampling"]["n_runs"] if config["experiment"]["resampling"]["enabled"] else 1
    retries = config["experiment"].get("retries", 3)
    timeout = config["experiment"].get("timeout", 3600)
    
    prompt_cfg = config["prompt"]
    system_prompt = prompt_cfg["system"]
    active_ds = config["active_dataset"]

    sampling_cfg = config["datasets"][active_ds].get("sampling")
    pair_lines = format_pair_lines(doc, active_ds, sampling_cfg=sampling_cfg)

    user_prompt = prompt_cfg["user_template"].format(
        doc_text=doc["doc_text"],
        pair_lines=pair_lines,
        doc_id=doc["id"],
    )
    if suffix := prompt_cfg.get("user_suffix", "").strip():
        user_prompt = user_prompt.rstrip() + "\n\n" + suffix

    # ── Set document context (needed by tools and similarity-based few-shot) ──
    ctx_token = CURRENT_DOC_ID.set(doc["id"])
    ctx_token_text = CURRENT_DOC_TEXT.set(doc["doc_text"])
    ctx_token_mentions = CURRENT_DOC_MENTIONS.set(frozenset(doc.get("mentions_map", {}).values()))
    ctx_token_prompt = CURRENT_USER_PROMPT.set(user_prompt)
    kfold_cfg = config["experiment"]["kfold"]
    n_folds = kfold_cfg["n_folds"] if kfold_cfg["enabled"] else 1
    ctx_token_fold = CURRENT_DOC_FOLD.set(doc["doc_idx"] % n_folds if n_folds > 1 else -1)

    # ── Few-shot systematic injection ────────────────────────────────────────
    fs_cfg = config.get("few_shot", {})
    few_shot_pairs = None
    if fs_cfg.get("enabled") and fs_cfg.get("systematic", True):
        few_shot_pairs = await get_few_shot_message_pairs(prompt_cfg["user_template"], active_ds, sampling_cfg=sampling_cfg)

    # ── Reprompt systematic injection ────────────────────────────────────────
    reprompt_str = ""
    if config.get("reprompt", {}).get("systematic", False):
        from tools.reprompt import reprompt as reprompt_tool
        reprompt_str = await reprompt_tool.ainvoke({})

    # 1. Run inference N times
    sampling_tasks = [
        run_inference_with_retry(graph_ainvoke, system_prompt, user_prompt, retries, few_shot_pairs, reprompt_str, doc_id=doc["id"], timeout=timeout)
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

        # Normalize pair order for binary undirected datasets
        ds_cfg = config["datasets"][active_ds]
        if ds_cfg.get("binary_undirected"):
            final_preds = [(min(s, t), lbl, max(s, t)) for s, lbl, t in final_preds]

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
        return {
            "id": doc["id"],
            "doc_idx": doc["doc_idx"],
            "lang": doc.get("lang", "unknown"),
            "gold_triples": doc["gold_triples"],
            "pred_triples": [],
            "pair_stats": {},
            "metrics": compute_ere_metrics(doc["gold_triples"], []),
            "context": {"system_prompt": "", "user_prompt": "", "raw_responses": [], "retry_failure": True},
        }
    finally:
        CURRENT_DOC_ID.reset(ctx_token)
        CURRENT_DOC_TEXT.reset(ctx_token_text)
        CURRENT_DOC_MENTIONS.reset(ctx_token_mentions)
        CURRENT_DOC_FOLD.reset(ctx_token_fold)


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

async def run_docs_concurrent(docs: List[Dict], config: Dict, graph_ainvoke, label: str = "") -> List[Dict]:
    """Run inference concurrently over a list of docs and return results."""
    semaphore = asyncio.Semaphore(config["experiment"].get("concurrency", 10))
    completed = 0
    total = len(docs)

    async def sem_task(doc):
        nonlocal completed
        async with semaphore:
            result = await process_document_resampled(doc, config, graph_ainvoke)
        completed += 1
        kfold_cfg_ = config["experiment"]["kfold"]
        n_folds = kfold_cfg_["n_folds"] if kfold_cfg_["enabled"] else 1
        fold_label = f"[fold {doc['doc_idx'] % n_folds + 1}/{n_folds}] " if n_folds > 1 else ""
        prefix = f"[{label}] " if label else ""
        print(f"{fold_label}{prefix}[{completed}/{total}] Processed doc {doc['id']}", flush=True)
        return result

    return await asyncio.gather(*[sem_task(doc) for doc in docs])


async def main():
    # 1. Load Config
    config = load_config()

    os.environ["LANGCHAIN_TRACING_V2"] = config["experiment"]["tracing"]
    os.environ["LANGCHAIN_PROJECT"] = config["experiment"]["tracing_name"]

    # Capture git state immediately so it reflects the code at launch time
    git_state = capture_git_state(Path.cwd())
    branch = git_state.get("branch", "unknown")
    commit = git_state.get("commit", "unknown")[:8]
    dirty = " (dirty)" if git_state.get("dirty") else ""
    print(f"Git: branch={branch}  commit={commit}{dirty}")

    # Dataset
    active_ds_key = config["active_dataset"]
    ds_config = config["datasets"][active_ds_key]
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

    kfold_cfg = config["experiment"].get("kfold", {})
    if kfold_cfg.get("enabled", False):
        await _run_kfold(config, ds_config, active_labels, graph_ainvoke, kfold_cfg, git_state=git_state)
    else:
        await _run_standard(config, ds_config, active_labels, graph_ainvoke, git_state=git_state)


async def _run_standard(config, ds_config, active_labels, graph_ainvoke, kfold_n_folds: int = 0, git_state=None):
    """Normal single-split evaluation. kfold_n_folds > 0 adds it to the report."""
    _logs_path, _stem, _ = make_run_stem("logs/allatonce", "run")
    trace_dump.TRACE_PATH = _logs_path / f"{_stem}.traces.jsonl.gz"
    trace_dump._sample_written = 0

    print(f"Engine started. Dataset: {ds_config['name']} | Samples: {ds_config['max_examples']} | Retries: {config['experiment'].get('retries', 0)}")

    if config.get("few_shot", {}).get("enabled"):
        print("Pre-loading few-shot training split...")
        await asyncio.to_thread(few_shot_preload)

    dataset_iter = load_hf_dataset_parsed(
        repo_id=ds_config["repo_id"],
        split=ds_config["split"],
        text_field=ds_config["text_field"],
        ann_field=ds_config["ann_field"],
        max_examples=ds_config.get("max_examples", 0),
        valid_labels=set(active_labels),
        binary_undirected=ds_config.get("binary_undirected", False),
    )

    docs = list(dataset_iter)
    results = await run_docs_concurrent(docs, config, graph_ainvoke)

    print("Aggregating results and computing metrics...")
    final_report = generate_run_report(
        results=results,
        total_processed_count=len(docs),
        config=config,
        valid_labels=set(active_labels),
    )

    if kfold_n_folds > 0:
        final_report["kfold_n_folds"] = kfold_n_folds

    print(f"Eval completed. Metric F1: {final_report['micro_f1']:.4f}")

    outfile = log_experiment(
        logdir="logs/allatonce",
        config=config,
        cli_args=sys.argv,
        results=final_report,
        filename_prefix="run",
        _stem=_stem,
        git_state=git_state,
    )
    print(f"Results logged to: {outfile}")
    for key in ["per_label", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "total_pairs", "binary"]:
        print(key, ":", final_report[key])
    if final_report.get("per_lang_metrics"):
        print("\nPer-language metrics:")
        for lang, lm in final_report["per_lang_metrics"].items():
            mc = lm["multiclass"]
            print(f"  [{lang}] pairs={lm['total_pairs']}  micro_f1={mc['micro_f1']:.4f}  macro_f1={mc['macro_f1']:.4f}  p={mc['micro_precision']:.4f}  r={mc['micro_recall']:.4f}")

    if config.get("mlflow", {}).get("enabled", False):
        mlflow_run_id = mlflow_log_run(
            config=config,
            final_report=final_report,
            outfile=outfile,
            trace_path=trace_dump.TRACE_PATH,
            run_name=_stem,
        )
        print(f"MLflow run: {mlflow_run_id}")


async def _run_kfold(config, ds_config, active_labels, graph_ainvoke, kfold_cfg, git_state=None):
    """K-fold CV: all docs processed in one parallel batch.
    Fold assignment (doc_idx % n_folds) is used only to filter few-shot examples.
    Output is identical to _run_standard plus kfold_n_folds in the report."""
    n_folds = kfold_cfg["n_folds"]
    print(f"K-fold mode: {n_folds} folds (doc_idx % {n_folds}) | dataset={ds_config['name']}")
    await _run_standard(config, ds_config, active_labels, graph_ainvoke, kfold_n_folds=n_folds, git_state=git_state)

if __name__ == "__main__":
    asyncio.run(main())