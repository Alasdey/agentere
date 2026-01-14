import os
import re
import json
import asyncio
from typing import List, Dict, Any, Tuple
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

# List of datasets to process
DATASETS = [
    {
        "repo_id": "Nofing/MAVEN-ERE-Causal-Events",
        "split": "test",
        "text_field": "text",
        "ann_field": "annots"
    },
    {
        "repo_id": "Nofing/MECI-v0.1-public-span",
        "split": "test",
        "text_field": "text",
        "ann_field": "annots"
    },
    {
        "repo_id": "Nofing/EventStoryLine-1.5-Causal",
        "split": "test", # Assuming test split exists
        "text_field": "text",
        "ann_field": "annots"
    }
]

# How many examples to process per dataset (for demo purposes)
MAX_EXAMPLES_PER_DATASET = 3 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def convert_triples_to_tool_pairs(gold_triples: List[tuple]) -> List[Dict[str, str]]:
    """
    Converts the (src, label, tgt) format from dataprep to the 
    [{"pair": "src,tgt", "label": "label"}] format required by the tool.
    """
    return [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]

def run_hf_coherence_agent():
    # 1. Setup the Model/Graph
    print("Initializing Agent...")
    graph, invoke, _ = build_chat_graph(
        model_id="mistralai/mistral-small-3.2-24b-instruct",
        # model_id="openai/gpt-5-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        tools=[coherence_check],
        enable_tools=True
    )

    print("=" * 80)
    print("RUNNING COHERENCE CHECK ON HF DATASETS")
    print("=" * 80)

    # 2. Iterate over Datasets
    for ds_config in DATASETS:
        repo_id = ds_config["repo_id"]
        split = ds_config["split"]
        
        print(f"\n{'='*80}")
        print(f"Dataset: {repo_id} (Split: {split})")
        print(f"{'='*80}")

        try:
            # Load dataset generator
            loader = load_hf_dataset_parsed(
                repo_id=repo_id,
                split=split,
                text_field=ds_config["text_field"],
                ann_field=ds_config["ann_field"],
                streaming=True,
                max_examples=MAX_EXAMPLES_PER_DATASET
            )

            for item in loader:
                doc_id = item.get('id', 'unknown')
                triples = item.get('gold_triples', [])
                
                # Skip documents with no relations
                if not triples:
                    print(f"🔍 Doc ID {doc_id}: Skipped (No relations found).")
                    continue

                print(f"🔍 Doc ID {doc_id}: Analyzing {len(triples)} relations...")

                # A. Convert data format for the tool
                pairs = convert_triples_to_tool_pairs(triples)

                # B. Construct Prompt
                # We only pass pairs. Rules are loaded from config.yaml inside the tool.
                system_prompt = "You are a data quality assistant."
                
                user_prompt = (
                    f"Analyze the relations from document {doc_id} for logical coherence.\n\n"
                    f"Here is the relation data. Call the coherence_check tool with these pairs:\n\n"
                    f"{json.dumps({'pairs': pairs}, indent=2)}"
                )

                # C. Invoke Graph
                try:
                    config = {"configurable": {"thread_id": f"{repo_id}-{doc_id}"}}
                    result = invoke(
                        messages=[
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt)
                        ],
                        config=config
                    )

                    # D. Display Results
                    # We print the last message which contains the summary
                    last_message = result["messages"][-1]
                    print(f"\n📝 Result:\n{last_message.content}")
                    print("-" * 80)

                except Exception as e:
                    print(f"❌ Error invoking agent for doc {doc_id}: {e}")

        except Exception as e:
            print(f"❌ Failed to load or process dataset {repo_id}: {e}")
            print("   (This might be due to incorrect split names or missing fields)")

if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable.")
    else:
        run_hf_coherence_agent()