import os
import re
import json
import yaml
from typing import List, Dict, Any

# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the centralized configuration file.
    """
    # Adjust path assuming script is run from project root or inside test_scripts
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# =============================================================================
# LANGSMITH TRACING CONFIGURATION
# =============================================================================

os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "Toto"

# =============================================================================
# IMPORTS
# =============================================================================
from dataprep.dataprep import load_hf_dataset_parsed
from tools.coherence import coherence_check
from model.model import build_chat_graph
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_triples_to_tool_pairs(gold_triples: List[tuple]) -> List[Dict[str, str]]:
    return [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]

def run_hf_coherence_agent():
    # 1. Load Configuration
    cfg = load_config()
    
    model_cfg = cfg.get("model", {})
    exp_cfg = cfg.get("experiment", {})
    
    # 2. Identify Active Dataset
    active_key = cfg.get("active_dataset")
    if not active_key:
        raise ValueError("No 'active_dataset' specified in config.yaml")
        
    all_datasets = cfg.get("datasets", {})
    if active_key not in all_datasets:
        raise ValueError(f"Active dataset key '{active_key}' not found in datasets config")
        
    ds_config = all_datasets[active_key]
    
    print(f"Initializing Agent for active dataset: {active_key} ({ds_config.get('name')})")

    # 3. Setup the Model/Graph
    graph, invoke, _ = build_chat_graph(
        model_id=model_cfg.get("default_model_id"),
        temperature=model_cfg.get("temperature", 0.0),
        base_url=model_cfg.get("base_url"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        tools=[coherence_check],
        enable_tools=exp_cfg.get("enable_tools", True)
    )

    print("=" * 80)
    print(f"RUNNING COHERENCE CHECK: {ds_config['name']}")
    print(f"Rule Set: {ds_config.get('rule_set')}")
    print(f"Max Examples: {ds_config.get('max_examples')}")
    print("=" * 80)

    # 4. Load and Process Dataset
    repo_id = ds_config["repo_id"]
    split = ds_config["split"]
    
    try:
        loader = load_hf_dataset_parsed(
            repo_id=repo_id,
            split=split,
            text_field=ds_config.get("text_field", "text"),
            ann_field=ds_config.get("ann_field", "annots"),
            streaming=True,
            # Hard-set max_examples from dataset config
            max_examples=ds_config.get("max_examples", 0)
        )

        for item in loader:
            doc_id = item.get('id', 'unknown')
            triples = item.get('gold_triples', [])
            
            if not triples:
                print(f"🔍 Doc ID {doc_id}: Skipped (No relations found).")
                continue

            print(f"🔍 Doc ID {doc_id}: Analyzing {len(triples)} relations...")

            # A. Convert data format
            pairs = convert_triples_to_tool_pairs(triples)

            # B. Construct Prompt
            system_prompt = exp_cfg.get("system_prompt", "You are a data quality assistant.")
            
            user_prompt = (
                f"Analyze the relations from document {doc_id} for logical coherence.\n\n"
                f"Here is the relation data. Call the coherence_check tool with these pairs:\n\n"
                f"{json.dumps({'pairs': pairs}, indent=2)}"
            )

            # C. Invoke Graph
            try:
                # Thread ID includes active key for trace clarity
                config = {"configurable": {"thread_id": f"{active_key}-{doc_id}"}}
                result = invoke(
                    messages=[
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ],
                    config=config
                )

                # D. Display Results
                last_message = result["messages"][-1]
                print(f"\n📝 Result:\n{last_message.content}")
                print("-" * 80)

            except Exception as e:
                print(f"❌ Error invoking agent for doc {doc_id}: {e}")

    except Exception as e:
        print(f"❌ Failed to load or process dataset {repo_id}: {e}")

if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable.")
    else:
        run_hf_coherence_agent()