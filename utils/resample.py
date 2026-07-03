# utils/resampling.py
from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Any, Dict

from utils.labels import NOREL, BINARY_POS, NOREL_VARIANTS as _NOREL_NORM

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
    num_runs = len(runs_outputs)

    for run in runs_outputs:
        for src, lab, tgt in run:
            pair_votes[(src, tgt)].append(lab)

    final_triples = []
    per_pair_stats = {}

    for (src, tgt) in pair_votes:
        votes = pair_votes[(src, tgt)]
        
        # Fill implicit NOREL for runs that didn't output this pair
        while len(votes) < num_runs:
            votes.append(NOREL)

        counts = Counter(votes)
        max_votes = max(counts.values())
        winners = [l for l, c in counts.items() if c == max_votes]

        if len(winners) == 1:
            final_label = winners[0]
        else:
            final_label = NOREL if tie_breaking == NOREL else random.choice(winners)
            
        if final_label.lower() not in _NOREL_NORM:
            final_triples.append((src, final_label, tgt))
            
        # Store stats for this pair (converted to dict for JSON serialization)
        key = f"{src},{tgt}"
        per_pair_stats[key] = {
            "votes": votes,
            "vote_counts": dict(counts),
            "final_label": final_label
        }

    return final_triples, per_pair_stats


def aggregate_votes_to_binary(
    pair_stats: Dict[str, Any],
    binary_default: str = "norel",
    novote_norel: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Recombines directed vote_counts (causes/causedby/norel) into a binary causal/norel
    decision per *undirected* pair, by summing causes+causedby votes against norel votes
    across both directions *before* picking a winner. Ties are broken by binary_default.

    A pair_list query is direction-specific: (src,tgt) and (tgt,src) are independent
    candidates that may each appear as their own key in pair_stats. Their vote_counts
    are merged here so that e.g. one direction predicting causal and the other predicting
    norel is a genuine 1-vs-1 tie, not an automatic causal via OR-of-directions.

    novote_norel: when True (default), a direction that was never queried/voted on at all
    counts as one implicit norel vote for that slot — so a lone causal vote with no
    opposing direction is also a tie (resolved by binary_default), not an outright win.
    When False, a missing direction contributes no vote (OR-of-directions: a single
    causal vote wins outright if the opposing direction was never queried).

    Pairs that resolve to norel are omitted (absence == norel, matching
    aggregate_run_triples' convention).
    """
    by_pair: Dict[frozenset, Dict[str, dict]] = defaultdict(dict)
    for key, stats in pair_stats.items():
        src, tgt = key.split(",", 1)
        by_pair[frozenset((src, tgt))][key] = stats["vote_counts"]

    result = []
    for pair, by_direction in by_pair.items():
        causal_votes = 0
        norel_votes = 0
        for vote_counts in by_direction.values():
            causal_votes += sum(c for lbl, c in vote_counts.items() if lbl.lower() not in _NOREL_NORM)
            norel_votes += sum(c for lbl, c in vote_counts.items() if lbl.lower() in _NOREL_NORM)
        if novote_norel and len(by_direction) < 2:
            norel_votes += 1

        if causal_votes > norel_votes:
            label = BINARY_POS
        elif norel_votes > causal_votes:
            label = NOREL
        else:
            label = BINARY_POS if binary_default == BINARY_POS else NOREL

        if label != NOREL:
            src, tgt = sorted(pair)
            result.append((src, label, tgt))
    return result