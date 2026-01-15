# utils/resampling.py
from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Any

def aggregate_run_triples(
    runs_outputs: List[List[Tuple[str, str, str]]],
    tie_breaking: str = "norel"
) -> List[Tuple[str, str, str]]:
    """
    Aggregates results from multiple sampling runs.
    """
    pair_votes = defaultdict(list)
    # Collect all unique pairs mentioned across all runs
    all_pairs = set()
    
    for run in runs_outputs:
        for src, lab, tgt in run:
            pair_votes[(src, tgt)].append(lab)
            all_pairs.add((src, tgt))
    
    # We must also account for the 'implicit NoRel' 
    # (If a pair was found in run A but not run B, run B voted 'NoRel')
    num_runs = len(runs_outputs)
    
    final_triples = []
    for (src, tgt) in all_pairs:
        votes = pair_votes[(src, tgt)]
        # Fill missing votes with NoRel
        while len(votes) < num_runs:
            votes.append("NoRel")
            
        counts = Counter(votes)
        max_votes = max(counts.values())
        winners = [l for l, c in counts.items() if c == max_votes]
        
        if len(winners) == 1:
            final_label = winners[0]
        else:
            final_label = "NoRel" if tie_breaking == "norel" else random.choice(winners)
            
        if final_label != "NoRel":
            final_triples.append((src, final_label, tgt))
            
    return final_triples