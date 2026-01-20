# utils/resampling.py
from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Any, Dict

def aggregate_run_triples(
    runs_outputs: List[List[Tuple[str, str, str]]],
    tie_breaking: str = "norel"
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """
    Aggregates results from multiple sampling runs.
    
    Returns:
        1. Final list of (src, label, tgt) triples for the winning predictions.
        2. A dictionary containing detailed stats per pair (for logging).
    """
    pair_votes = defaultdict(list)
    # Collect all unique pairs mentioned across all runs
    all_pairs = set()

    num_runs = len(runs_outputs)
    
    # 1. Collect all explicit predictions
    for run in runs_outputs:
        for src, lab, tgt in run:
            pair_votes[(src, tgt)].append(lab)
            all_pairs.add((src, tgt))
    
    # We must also account for the 'implicit NoRel' 
    # (If a pair was found in run A but not run B, run B voted 'NoRel')
    num_runs = len(runs_outputs)
    
    final_triples = []
    
    # Stats container: Key = "src,tgt" string (to match log format)
    per_pair_stats = {}

    # 2. Iterate over all seen pairs to determine winners
    for (src, tgt) in all_pairs:
        votes = pair_votes[(src, tgt)]
        
        # Fill implicit "NoRel" for runs that didn't output this pair
        while len(votes) < num_runs:
            votes.append("NoRel")
            
        counts = Counter(votes)
        max_votes = max(counts.values())
        winners = [l for l, c in counts.items() if c == max_votes]
        
        if len(winners) == 1:
            final_label = winners[0]
        else:
            final_label = "NoRel" if tie_breaking == "norel" else random.choice(winners)
            
        if final_label not in ("NoRel", "None", "Unknown"):
            final_triples.append((src, final_label, tgt))
            
        # Store stats for this pair (converted to dict for JSON serialization)
        key = f"{src},{tgt}"
        per_pair_stats[key] = {
            "votes": votes,
            "vote_counts": dict(counts),
            "final_label": final_label
        }
            
    return final_triples, per_pair_stats