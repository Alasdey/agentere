
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
    "initial_samples": 6,
    "sample_increment": 2,
    "max_cycles": 100,
    # Number of error cases to analyze per parent candidate
    "err_agg": 20, 
    # How many critiques to bundle into one rewriting request
    "critic_batch_size": 6, 
    "threshold": 1.0,
    "max_samples": 714,
    "meta_llm": "deepseek/deepseek-v3.2",
    "meta_temp": 0.6,
}

# Revised template to match main.py structure
CONSTANT_USER_TEMPLATE = """
Task: Extract causal relations from the text below.

Text:
{doc_text}

Pairs to classify (use EXACT pair ids; output order does NOT matter):
{pair_lines}

Output a JSON array of objects with "pair" and "label" at the end of your answer.
Example: [{{"pair": "e1,e2", "label": "CAUSE"}}]
If there are no mentions in the text and no pairs to classify return the example.
"""

# =============================================================================
# HELPERS
# =============================================================================

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
        
        
        # Load fresh dataset slice
        self.full_dataset = list(load_hf_dataset_parsed(
            repo_id=self.ds_cfg["repo_id"],
            split=self.ds_cfg["split"],
            text_field=self.ds_cfg["text_field"],
            ann_field=self.ds_cfg["ann_field"],
            max_examples=OPTIM_CONFIG["max_samples"],
            streaming=True 
        ))

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

        # Prepare docs: generate the 'pair_lines' locally exactly as in main.py
        processed_docs = []
        for doc in dataset:
            d = copy.deepcopy(doc)
            
            # --- Logic from main.py ---
            mentions_map = d.get("mentions_map", {})

            ###########################
            # print(doc, mentions_map)

            formatted_pairs = []
            
            # We assume gold_triples exist since we strictly loaded them via dataprep
            for src, lbl, tgt in d["gold_triples"]:
                src_quote = mentions_map.get(src, "")
                tgt_quote = mentions_map.get(tgt, "")
                # Format: "T0 ("shooting"), T1 ("trial")"
                formatted_pairs.append(f"{src} (\"{src_quote}\"), {tgt} (\"{tgt_quote}\")")
            
            ##########################
            # print(formatted_pairs)

            d["pair_lines"] = "\n".join(formatted_pairs)
            # --------------------------
            
            processed_docs.append(d)

        tasks = [process_document_resampled(d, eval_config, ds_copy, engine_ainvoke) for d in processed_docs]
        raw_results = await asyncio.gather(*tasks)
        ############################
        # print("\nraw_results[0]", raw_results[0])
        valid = [r for r in raw_results if r is not None]
        tp, fp, fn, support = 0, 0, 0, 0
        
        for r in valid:
            # 1. Extract raw lists
            gold_raw = r.get("gold_triples", [])
            pred_raw = r.get("pred_triples", [])

            # 2. Filter out "NoRel" from both GOLD and PRED case-insensitively
            gold_set = set(
                (t[0], t[1], t[2]) 
                for t in gold_raw 
                if t[1].lower() != "norel"
            )
            pred_set = set(
                (t[0], t[1], t[2]) 
                for t in pred_raw 
                if t[1].lower() != "norel"
            )

            # 3. Calculate metrics based on strict set operations
            _tp = len(gold_set.intersection(pred_set))
            _fp = len(pred_set - gold_set)
            _fn = len(gold_set - pred_set)
            
            tp += _tp
            fp += _fp
            fn += _fn
            support += len(gold_set)

        candidate.support = support
        candidate.tp, candidate.fp, candidate.fn = tp, fp, fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        candidate.score = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        candidate.results = valid

    async def _analyze_single_failure_map(self, r: Dict, current_prompt: str, meta_llm) -> str:
        """
        MAP STEP: Analyze a single failure instance.
        """
        clean_gold = [t for t in r['gold_triples'] if t[1].lower() != "norel"]
        clean_pred = [t for t in r['pred_triples'] if t[1].lower() != "norel"]
        
        # Skip if no interesting errors
        if str(clean_gold) == str(clean_pred):
            return ""

        micro_prompt = f"""
        You are diagnosing an error in an Event Relation Extraction system.
        
        INPUT TEXT:
        {r['context']['user_prompt']}

        EXPECTED OUTPUT (GOLD):
        {triples_to_json_format(clean_gold)}

        ACTUAL OUTPUT (PRED):
        {triples_to_json_format(clean_pred)}

        Briefly diagnose why the system prompt failed here. 
        Focus on specific language patterns, missing definitions, or edge cases.
        Be concise (1 sentence). Start with "Error:".
        """
        try:
            resp = await meta_llm([HumanMessage(content=micro_prompt)])
            return resp['messages'][-1].content.strip()
        except Exception:
            return ""

    async def _rewrite_prompt_from_critiques(self, current_prompt: str, critiques: List[str], meta_llm) -> str:
        """
        Takes a batch of critiques and rewrites the prompt to address them.
        """
        if not critiques:
            return None
            
        combined_critiques = "\n".join([f"- {c}" for c in critiques])
        
        rewrite_prompt = f"""
        You are an expert Prompt Engineer.
        
        CURRENT SYSTEM PROMPT:
        \"\"\"{current_prompt}\"\"\"

        OBSERVED FAILURES (Critiques):
        {combined_critiques}

        TASK:
        Rewrite the CURRENT SYSTEM PROMPT to fix the issues mentioned in the critiques.
        
        GUIDELINES:
        1. **Fix the specific logic** mentioned in the critiques (e.g., clarify definitions, adjust constraints).
        2. **Remove inconsistencies** Remove sections that are inconsistent, half written or redundant. 
        3. **Preserve** all other parts of the prompt that are not related to these errors. Do not rewrite sections that are independant from the critiques.
        4. Ensure the new prompt remains coherent and maintains the original output format instructions.
        5. Output ONLY the new system prompt. No markdown, no conversational fillers.
        """
        
        try:
            resp = await meta_llm([HumanMessage(content=rewrite_prompt)])
            new_prompt = resp["messages"][-1].content.strip()
            
            # Basic cleanup
            if new_prompt.startswith("```"):
                new_prompt = new_prompt.replace("```markdown", "").replace("```", "").strip()
            
            return new_prompt
        except Exception as e:
            print(f"Rewrite failed: {e}")
            return None

    async def audit_and_evolve(self, parent: PromptCandidate, cycle: int) -> List[PromptCandidate]:
        """
        Uses Map-Rewrite strategy:
        1. Map: Diagnose errors individually.
        2. Batch: Group critiques.
        3. Rewrite: LLM fixes prompt based on batch.
        """
        _, _, meta_llm = build_chat_graph(
            model_id=OPTIM_CONFIG["meta_llm"],
            temperature=OPTIM_CONFIG["meta_temp"],
            base_url=self.config["model"]["base_url"]
        )

        # Get a mix of errors (Precision vs Recall)
        # print("\nparent", parent)
        errors = [r for r in parent.results if r["metrics"]["f1"] < OPTIM_CONFIG["threshold"]]
        random.shuffle(errors)
        
        selection = errors[:OPTIM_CONFIG["err_agg"]]
        if not selection:
            return []

        print(f"    ...Auditing {len(selection)} error cases...")

        # 2. MAP: Generate critiques
        map_tasks = [
            self._analyze_single_failure_map(r, parent.system_prompt, meta_llm)
            for r in selection
        ]
        raw_critiques = await asyncio.gather(*map_tasks)
        valid_critiques = [c for c in raw_critiques if c]

        if not valid_critiques:
            return []

        # 3. BATCH & REWRITE
        generated_candidates = []
        batch_size = OPTIM_CONFIG["critic_batch_size"]
        
        # Create batches (e.g. if we have 6 critiques and batch size 3, we get 2 batches)
        # We can also create overlapping/random subsets if desired, but chunks work well for coverage.
        batches = [valid_critiques[i:i + batch_size] for i in range(0, len(valid_critiques), batch_size)]
        
        rewrite_tasks = [
            self._rewrite_prompt_from_critiques(parent.system_prompt, batch, meta_llm)
            for batch in batches
        ]
        
        # Run rewrites in parallel
        new_prompts = await asyncio.gather(*rewrite_tasks)

        for new_val in new_prompts:
            if new_val and new_val != parent.system_prompt:
                generated_candidates.append(PromptCandidate(new_val, cycle))

        return generated_candidates

    async def run(self):
        samples = OPTIM_CONFIG["initial_samples"]
        top_scores = []

        for cycle in range(OPTIM_CONFIG["max_cycles"]):
            print(f"\n--- CYCLE {cycle} (Samples: {samples}) ---")
            
            # Load fresh dataset slice
            dataset = random.sample(self.full_dataset, samples)
            # dataset = list(load_hf_dataset_parsed(
            # repo_id=self.ds_cfg["repo_id"],
            # split=self.ds_cfg["split"],
            # text_field=self.ds_cfg["text_field"],
            # ann_field=self.ds_cfg["ann_field"],
            # max_examples=samples,
            # streaming=True 
            # ))

            ########################
            # print(dataset[0])

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
            
            print("len(elites):", len(elites))
            if len(elites) > 0:
                print("\nelites[0]", elites[0])
                print(f"Top Score: {elites[0].score:.4f} (TP: {elites[0].tp}, Support: {elites[0].support})")
                top_scores.append(elites[0].score)
                print(f"Historic of scores:", top_scores)

            # Evolve
            evo_tasks = [self.audit_and_evolve(p, cycle) for p in elites]
            evo_results = await asyncio.gather(*evo_tasks)
            
            new_pop = copy.deepcopy(elites)
            for res in evo_results:
                new_pop.extend(res)
            
            print("len(new_pop):", len(new_pop))
            while len(new_pop) < OPTIM_CONFIG["max_prompts"]:
                if len(elites) > 0:
                    new_pop.append(PromptCandidate(random.choice(elites).system_prompt, cycle))
                else:
                    break # Should not happen if population initialized
            
            self.population = new_pop[:OPTIM_CONFIG["max_prompts"]]
            samples += OPTIM_CONFIG["sample_increment"]
            
            if len(self.population) > 0:
                print(f"\n\nCURRENT BEST PROMPT:\n{self.population[0].system_prompt}")

        if len(self.population) > 0:
            print(f"\n\nFINAL MHEH PROMPT:\n{self.population[1].system_prompt}")
            print(f"\n\nFINAL PROMPT:\n{self.population[0].system_prompt}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    asyncio.run(EvolutionaryOptimizer(cfg).run())