import asyncio
import yaml
import copy
import random
import re
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from main import process_document_resampled, build_chat_graph
from dataprep.dataprep import load_hf_dataset_parsed

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
OPTIM_CONFIG = {
    "max_prompts": 8,       # Population size
    "n_selected": 2,       # Elites to keep and use as parents
    "initial_samples": 5,  # Starting documents for evaluation
    "sample_increment": 5, # How many docs to add per cycle
    "max_cycles": 10,
    "eval_concurrency": 10
}

class PromptCandidate:
    def __init__(self, system_prompt: str, version: int = 0):
        self.system_prompt = system_prompt
        self.version = version
        self.score = 0.0
        self.results = []

    def __repr__(self):
        return f"[V{self.version}] F1: {self.score:.4f}"

class EvolutionaryPromptOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ds_cfg = config["datasets"][config["active_dataset"]]
        # Initialize population with the base prompt from config
        initial_prompt = config["prompts"][self.ds_cfg["prompt"]]["system"]
        self.population = [PromptCandidate(initial_prompt) for _ in range(OPTIM_CONFIG["max_prompts"])]
        
        # Tools and Graph setup
        from tools import get_enabled_tools
        self.tools = get_enabled_tools(config["experiment"].get("tools", []))
        
    async def evaluate_population(self, n_samples: int):
        """Asynchronously evaluates all prompts in the population."""
        # Load dataset slice
        dataset = list(load_hf_dataset_parsed(
            repo_id=self.ds_cfg["repo_id"],
            split=self.ds_cfg["split"],
            max_examples=n_samples
        ))

        _, _, graph_ainvoke = build_chat_graph(
            model_id=self.config["model"]["default_model_id"],
            temperature=self.config["model"]["temperature"],
            base_url=self.config["model"]["base_url"],
            tools=self.tools,
            enable_tools=self.config["experiment"]["enable_tools"]
        )

        async def eval_candidate(candidate: PromptCandidate):
            # Temporarily override config prompt for this call
            local_config = copy.deepcopy(self.config)
            local_config["prompts"]["temp_optim"] = {
                "system": candidate.system_prompt,
                "user_template": self.config["prompts"][self.ds_cfg["prompt"]]["user_template"]
            }
            local_config["datasets"][local_config["active_dataset"]]["prompt"] = "temp_optim"
            
            # Execute inference for all docs in subset
            tasks = [process_document_resampled(doc, local_config, self.ds_cfg, graph_ainvoke) for doc in dataset]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            
            # Calculate Micro-F1
            tp, fp, fn = 0, 0, 0
            for r in valid_results:
                m = r["metrics"]
                tp += m.get("tp", 0)
                # Recalculate precision/recall components for global micro-F1
                fp += (m["predicted"] - m["tp"])
                fn += (m["support"] - m["tp"])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            candidate.score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            candidate.results = valid_results

        await asyncio.gather(*(eval_candidate(c) for c in self.population))

    async def audit_and_evolve(self, parent: PromptCandidate, cycle: int) -> List[PromptCandidate]:
        """Diagnostic Audit & Pruning via LLM."""
        _, _, graph_ainvoke = build_chat_graph(
            model_id="openai/gpt-4o", # Use a strong model for meta-reasoning
            temperature=0.5
        )

        # Build error context from parent's failed samples
        error_context = ""
        for res in parent.results[:3]: # Look at first 3 results for context
            error_context += f"\nDoc ID: {res['id']}\nGold: {res['gold_triples']}\nPred: {res['pred_triples']}\n"

        audit_prompt = f"""
        # Task: Instructional Refinement
        You are an expert prompt engineer. Analyze the performance of an ERE extraction prompt.
        
        # Current Prompt:
        {parent.system_prompt}
        
        # Sample Error Analysis:
        {error_context}

        # Objective:
        1. PRUNE: Identify specific paragraphs or rules in the Current Prompt that are causing false positives or logic errors.
        2. AUGMENT: Create 2-3 new 'Instructional Chunks' to fix these errors.
        
        # Output Format:
        [PRUNE_START] text to remove [PRUNE_END]
        [CHUNK_START] new instruction [CHUNK_END]
        """
        
        msg = await graph_ainvoke([HumanMessage(content=audit_prompt)])
        content = msg["messages"][-1].content

        # Simple regex parsing for evolution
        prune_match = re.search(r"\[PRUNE_START\](.*?)\[PRUNE_END\]", content, re.DOTALL)
        chunks = re.findall(r"\[CHUNK_START\](.*?)\[CHUNK_END\]", content, re.DOTALL)

        pruned_base = parent.system_prompt
        if prune_match:
            pruned_base = pruned_base.replace(prune_match.group(1).strip(), "")

        new_variants = []
        for chunk in chunks:
            new_variants.append(PromptCandidate(pruned_base + "\n" + chunk.strip(), version=cycle))
        
        return new_variants

    async def run_optimization(self):
        current_samples = OPTIM_CONFIG["initial_samples"]

        for cycle in range(OPTIM_CONFIG["max_cycles"]):
            print(f"\n--- Cycle {cycle} (Samples: {current_samples}) ---")
            
            # 1. Evaluate
            await self.evaluate_population(current_samples)
            
            # Sort by score descending
            self.population.sort(key=lambda x: x.score, reverse=True)
            elites = self.population[:OPTIM_CONFIG["n_selected"]]
            
            avg_f1 = sum(c.score for c in self.population) / len(self.population)
            print(f"Cycle Result: Best F1: {elites[0].score:.4f} | Avg F1: {avg_f1:.4f}")
            print(f"Top Prompt Version: {elites[0].version}")

            # 2. Evolve (Audit & Prune)
            new_population = copy.deepcopy(elites) # Elitism: Preserved as-is
            
            evolution_tasks = [self.audit_and_evolve(parent, cycle) for parent in elites]
            variant_groups = await asyncio.gather(*evolution_tasks)
            
            for group in variant_groups:
                new_population.extend(group)

            # 3. Fill remaining slots
            while len(new_population) < OPTIM_CONFIG["max_prompts"]:
                parent = random.choice(elites)
                new_population.append(PromptCandidate(parent.system_prompt, version=cycle))

            self.population = new_population[:OPTIM_CONFIG["max_prompts"]]
            current_samples += OPTIM_CONFIG["sample_increment"]

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    optimizer = EvolutionaryPromptOptimizer(config)
    asyncio.run(optimizer.run_optimization())