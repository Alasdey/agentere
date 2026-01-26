import asyncio
import yaml
import copy
import random
import re
import json
import time
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from main import process_document_resampled, build_chat_graph
from dataprep.dataprep import load_hf_dataset_parsed

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
OPTIM_CONFIG = {
    "max_prompts": 8,           
    "n_selected": 2,           
    "initial_samples": 10,      
    "sample_increment": 2,     
    "max_cycles": 10,
}

# Revised template: Instead of "Pairs to classify", we provide "Events present"
# This mirrors how MAVEN-ERE works (finding relations among present events)
CONSTANT_USER_TEMPLATE = """
Task: Extract causal relations from the text below.
Only use the Event IDs provided in the text (e.g., e0, e1).

Text:
{doc_text}

List of all events mentioned in the text:
{pair_lines}

Output ONLY a JSON array of objects with "pair" and "label".
Example: [{"pair": "e1,e2", "label": "CAUSE"}]
"""

# =============================================================================
# HELPERS
# =============================================================================

def get_event_ids_from_text(text: str) -> str:
    """Finds all instances of <eID ...> in text and returns a unique list of IDs."""
    # Matches <e123 or <T123
    ids = re.findall(r"<([a-zA-Z0-9]+)\s+", text)
    unique_ids = sorted(list(set(ids)), key=lambda x: (len(x), x))
    return ", ".join(unique_ids)

def triples_to_json_format(triples: List[Tuple[str, str, str]]) -> str:
    return json.dumps([{"pair": f"{t[0]},{t[2]}", "label": t[1]} for t in triples])

class PromptCandidate:
    def __init__(self, system_prompt: str, version: int = 0):
        self.id = f"v{version}-{random.getrandbits(16):04x}"
        self.system_prompt = system_prompt
        self.score = 0.0
        self.tp, self.fp, self.fn, self.support = 0, 0, 0, 0
        self.results = []

# =============================================================================
# OPTIMIZER
# =============================================================================

class EvolutionaryOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ds_key = config["active_dataset"]
        self.ds_cfg = config["datasets"][self.ds_key]
        
        # Load initial prompt from config
        initial_sys = config["prompts"][self.ds_cfg["prompt"]]["system"]
        self.population = [PromptCandidate(initial_sys) for _ in range(OPTIM_CONFIG["max_prompts"])]

    async def evaluate_candidate(self, candidate: PromptCandidate, dataset: List[Dict], engine_ainvoke):
        """Evaluates a prompt against the dataset."""
        print(f"  [EVAL] Testing {candidate.id}...")
        
        # Setup local config for this candidate
        eval_config = copy.deepcopy(self.config)
        eval_config["prompts"]["evolved"] = {
            "system": candidate.system_prompt,
            "user_template": CONSTANT_USER_TEMPLATE
        }
        
        # Force the framework to run this candidate
        ds_copy = copy.deepcopy(self.ds_cfg)
        ds_copy["prompt"] = "evolved"

        # Prepare docs: generate the 'pair_lines' locally as a list of IDs for the prompt
        processed_docs = []
        for doc in dataset:
            d = copy.deepcopy(doc)
            # This ensures {pair_lines} in CONSTANT_USER_TEMPLATE is a list of IDs
            d["pair_lines"] = get_event_ids_from_text(doc["doc_text"])
            processed_docs.append(d)

        tasks = [process_document_resampled(d, eval_config, ds_copy, engine_ainvoke) for d in processed_docs]
        raw_results = await asyncio.gather(*tasks)
        
        valid = [r for r in raw_results if r is not None]
        tp, fp, fn, support = 0, 0, 0, 0
        
        for r in valid:
            m = r["metrics"]
            tp += m["tp"]
            fp += (m["predicted"] - m["tp"])
            fn += (m["support"] - m["tp"])
            support += m["support"]

        candidate.support = support
        candidate.tp, candidate.fp, candidate.fn = tp, fp, fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        candidate.score = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        candidate.results = valid

    async def audit_and_evolve(self, parent: PromptCandidate, cycle: int) -> List[PromptCandidate]:
        """Uses LLM to prune and improve the system prompt based on errors."""
        _, _, meta_llm = build_chat_graph(
            model_id="deepseek/deepseek-v3.2", # Use a strong model for meta-analysis
            temperature=0.6,
            base_url=self.config["model"]["base_url"]
        )

        # Get a mix of errors (Precision vs Recall)
        errors = [r for r in parent.results if r["metrics"]["f1"] < 0.9]
        random.shuffle(errors)
        example_str = ""
        for r in errors[:2]:
            example_str += f"\nTEXT: {r['id']}\nGOLD: {triples_to_json_format(r['gold_triples'])}\nPRED: {triples_to_json_format(r['pred_triples'])}\n"

        meta_prompt = f"""
        Analyze this Event Relation Extraction prompt. 
        Current Prompt: {parent.system_prompt}
        
        Errors observed:
        {example_str}
        
        1. Identify the 'harmful' instruction causes the error (PRUNE).
        2. Create 2 refined instructions to improve logic (AUG).
        
        Output format:
        PRUNE: <text to delete>
        AUG1: <new rule 1>
        AUG2: <new rule 2>
        """
        
        try:
            resp = await meta_llm([HumanMessage(content=meta_prompt)])
            text = resp["messages"][-1].content
            
            p_match = re.search(r"PRUNE:\s*(.*)", text)
            augs = re.findall(r"AUG\d:\s*(.*)", text)
            
            base = parent.system_prompt
            if p_match and "none" not in p_match.group(1).lower():
                base = base.replace(p_match.group(1).strip(), "")
                
            return [PromptCandidate(base + "\n" + a.strip(), cycle) for a in augs]
        except:
            return []

    async def run(self):
        samples = OPTIM_CONFIG["initial_samples"]
        
        for cycle in range(OPTIM_CONFIG["max_cycles"]):
            print(f"\n--- CYCLE {cycle} (Samples: {samples}) ---")
            
            # Load fresh dataset slice
            dataset = list(load_hf_dataset_parsed(
                repo_id=self.ds_cfg["repo_id"], split=self.ds_cfg["split"], max_examples=samples
            ))

            _, _, engine = build_chat_graph(
                model_id=self.config["model"]["default_model_id"],
                temperature=0.0,
                base_url=self.config["model"]["base_url"]
            )

            # Evaluate all
            await asyncio.gather(*(self.evaluate_candidate(c, dataset, engine) for c in self.population))
            
            # Select
            self.population.sort(key=lambda x: x.score, reverse=True)
            elites = self.population[:OPTIM_CONFIG["n_selected"]]
            
            print(f"Top Score: {elites[0].score:.4f} (TP: {elites[0].tp}, Support: {elites[0].support})")

            # Evolve
            evo_tasks = [self.audit_and_evolve(p, cycle) for p in elites]
            evo_results = await asyncio.gather(*evo_tasks)
            
            new_pop = copy.deepcopy(elites)
            for res in evo_results:
                new_pop.extend(res)
            
            while len(new_pop) < OPTIM_CONFIG["max_prompts"]:
                new_pop.append(PromptCandidate(random.choice(elites).system_prompt, cycle))
            
            self.population = new_pop[:OPTIM_CONFIG["max_prompts"]]
            samples += OPTIM_CONFIG["sample_increment"]

        print(f"\nFINAL PROMPT:\n{self.population[0].system_prompt}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    asyncio.run(EvolutionaryOptimizer(cfg).run())