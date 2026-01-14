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
from tools.coherence import coherence_check
from model.model import build_chat_graph
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================================================
# DATA MOCKING (From your provided samples)
# =============================================================================

MAVEN_ERE_ROW = {
    "id": "ce42b57b90199c73b1bc5c1b46cd9f0b",
    "annots": "<e15 headed> PRECONDITION <e16 freed>, <e1 fought> PRECONDITION <e15 headed>, <e1 fought> PRECONDITION <e16 freed>",
    "text": "Battle of Saint Gotthard..."
}

MECI_ROW = {
    "id": "economic_crisis-week4-isik-41860_chunk_5.ann",
    "annots": "<T1 ölümünden> CauseEffect <T2 oldu>, <T2 oldu> CauseEffect <T7 oldu>, <T2 oldu> CauseEffect <T4 etti>, <T2 oldu> EffectCause <T1 ölümünden>, <T7 oldu> EffectCause <T2 oldu>, <T4 etti> EffectCause <T2 oldu>, <T0 ulaştı> NoRel <T1 ölümünden>, <T0 ulaştı> NoRel <T4 etti>, <T1 ölümünden> NoRel <T6 uzaklaştırdı>, <T2 oldu> NoRel <T0 ulaştı>, <T4 etti> NoRel <T0 ulaştı>, <T4 etti> NoRel <T5 pekiştirdi>, <T4 etti> NoRel <T6 uzaklaştırdı>, <T5 pekiştirdi> NoRel <T7 oldu>, <T6 uzaklaştırdı> NoRel <T0 ulaştı>, <T6 uzaklaştırdı> NoRel <T4 etti>, <T6 uzaklaştırdı> NoRel <T5 pekiştirdi>, <T7 oldu> NoRel <T4 etti>,",
    "text": "1954 yılında..."
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Borrowing the regex from dataprep/dataprep.py for consistency
RELATION_TRIPLE_RE = re.compile(
    r"<([^>\s]+)\s+([^>\n]+)>"       # Source Tag: <ID Content>
    r"\s+"                           # Space
    r"([A-Za-z0-9_\-]+)"             # Relation Label
    r"\s+"                           # Space
    r"<([^>\s]+)\s+([^>\n]+)>"       # Target Tag: <ID Content>
)

def parse_annots_to_pairs(annots_str: str) -> List[Dict[str, str]]:
    """
    Converts the raw annotation string into the format expected by the coherence tool.
    Format: [{"pair": "ID1,ID2", "label": "REL"}, ...]
    """
    pairs = []
    if not annots_str:
        return pairs
    
    for match in RELATION_TRIPLE_RE.finditer(annots_str):
        src_id = match.group(1)
        label = match.group(3)
        tgt_id = match.group(4)
        
        pairs.append({
            "pair": f"{src_id},{tgt_id}",
            "label": label
        })
        
    return pairs

def run_coherence_agent():
    # 1. Setup the Model/Graph
    graph, invoke, _ = build_chat_graph(
        # model_id="deepseek/deepseek-r1",
        model_id="mistralai/mistral-small-3.2-24b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        tools=[coherence_check], # The updated tool
        enable_tools=True
    )

    datasets = [
        {"name": "MavenERE", "data": MAVEN_ERE_ROW},
        {"name": "Meci", "data": MECI_ROW}
    ]

    print("=" * 80)
    print("RUNNING COHERENCE CHECK AGENT (Config-based)")
    print("=" * 80)

    for dataset_info in datasets:
        name = dataset_info["name"]
        row = dataset_info["data"]
        
        print(f"\n🔍 Processing Dataset: {name}")
        
        # A. Parse Raw Data
        pairs = parse_annots_to_pairs(row["annots"])
        
        # B. Construct Prompt
        # NOTE: We NO LONGER pass 'rules' in the JSON. 
        # We strictly pass the 'pairs'. The tool handles the rest.
        system_prompt = "You are a data quality assistant."
        
        user_prompt = (
            f"Analyze the relations from the {name} dataset for logical coherence.\n\n"
            f"Here is the relation data. Call the coherence_check tool with these pairs:\n\n"
            f"{json.dumps({'pairs': pairs}, indent=2)}\n\n"
            f"Please report the summary of incoherence provided by the tool."
        )

        # C. Invoke Graph
        try:
            config = {"configurable": {"thread_id": f"coherence-{name}"}}
            result = invoke(
                messages=[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ],
                config=config
            )

            # D. Display Results
            for msg in result["messages"]:
                # Print the content directly, which will be the human summary from the tool
                print(msg.content)
                
            print("-" * 80)

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable.")
    else:
        run_coherence_agent()