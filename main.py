# FILE: main_eval.py
import os
import json
import asyncio
import re
from typing import List, Dict, Any


os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "Toto"

from dataprep.dataprep import load_hf_dataset_parsed
from model.model import build_chat_graph
from tools.coherence import coherence_check
from utils.metrics import compute_ere_metrics
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Configuration ---
REPOS = ["Nofing/MAVEN-ERE-Causal-Events"]
MAX_EXAMPLES = 10
CONCURRENCY = 10

def extract_json_array(messages: List[Any]) -> List[tuple]:
    """Helper to find the final triple list in assistant response."""
    triples = []
    # Look for a JSON block or a list format
    # Simple regex to find (src, label, tgt) pattern in text if JSON fails
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            # Look for JSON array in markdown
            match = re.search(r"\[\s*\{.*?\}\s*\]", m.content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return [(obj['pair'].split(',')[0], obj['label'], obj['pair'].split(',')[1]) for obj in data]
                except:
                    pass
    return triples

async def evaluate_document(item, ainvoke_func, sem):
    async with sem:
        doc_id = item["id"]
        gold = item["gold_triples"]
        
        # Prepare the setup for the Agent
        # We give the agent the text and ask it to predict AND check coherence
        sys_msg = SystemMessage("You are an expert Causal Relation Annotator. Output ONLY valid JSON.")
        prompt = (
            f"Text: {item['doc_text']}\n\n"
            "Task: Identify all causal relations between marked event triggers. "
            "1. Predict the relations.\n"
            "2. Use the coherence_check tool to validate your predictions.\n"
            "3. If violations are found, correct your labels.\n"
            "Final Answer: Return a JSON array: [{\"pair\": \"T1,T2\", \"label\": \"PRECONDITION\"}]"
        )
        
        try:
            print("llm called")
            res = await ainvoke_func(
                [sys_msg, HumanMessage(prompt)],
                config={"configurable": {"thread_id": f"eval_{doc_id}"}}
            )
            preds = extract_json_array(res["messages"])
            metrics = compute_ere_metrics(gold, preds)
            print(metrics)
            return {"id": doc_id, "metrics": metrics, "gold_count": len(gold), "pred_count": len(preds)}
        except Exception as e:
            print(f"Error on {doc_id}: {e}")
            return None

async def main():
    # 1. Setup
    graph, _, ainvoke = build_chat_graph(
        model_id="openai/gpt-5-mini",
        tools=[coherence_check],
        enable_tools=True
    )
    
    sem = asyncio.Semaphore(CONCURRENCY)
    loader = load_hf_dataset_parsed(
        repo_id=REPOS[0],
        max_examples=MAX_EXAMPLES,
        streaming=True
    )
    
    # 2. Run async tasks
    tasks = [evaluate_document(item, ainvoke, sem) for item in loader]
    results = await asyncio.gather(*tasks)
    
    # 3. Aggregate
    valid_results = [r for r in results if r]
    avg_f1 = sum(r['metrics']['f1'] for r in valid_results) / len(valid_results)
    avg_prec = sum(r['metrics']['precision'] for r in valid_results) / len(valid_results)
    avg_rec = sum(r['metrics']['recall'] for r in valid_results) / len(valid_results)
    
    print("\n" + "="*30)
    print("EVALUATION COMPLETE")
    print("="*30)
    print(f"Documents: {len(valid_results)}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")

if __name__ == "__main__":
    asyncio.run(main())