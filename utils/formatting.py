from __future__ import annotations

import itertools
import json
import random
from typing import List, Optional, Tuple

_NOREL_LABELS = frozenset({"norel", "none", "no_rel"})

# Datasets that use the binary-undirected sampled-pair-list format
_BINARY_SAMPLED_DATASETS = frozenset({"event_story_line", "maven_ere"})

# Datasets that use pair_list_ids from HF directly (undirected deduplication for causal_timebank)
_PAIR_LIST_DATASETS = frozenset({"meci", "causal_timebank", "causal_timebank_standard"})


def convert_to_binary_undirected(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Collapse directional labels to CAUSAL/NoRel and deduplicate by unordered pair."""
    seen: set = set()
    result = []
    for src, lbl, tgt in triples:
        key = frozenset([src, tgt])
        if key in seen:
            continue
        seen.add(key)
        canonical = tuple(sorted([src, tgt]))
        label = "NoRel" if lbl.lower() in _NOREL_LABELS else "CAUSAL"
        result.append((canonical[0], label, canonical[1]))
    return result


def _sample_pair_lines(doc: dict, sampling_cfg: dict) -> str:
    """Build a sampled pair list: all positive pairs + a random NoRel fraction."""
    mentions_map = doc.get("mentions_map", {})
    mentions = sorted(mentions_map.keys())
    positive = {
        (min(s, t), max(s, t))
        for s, lbl, t in doc["gold_triples"]
        if lbl.lower() not in _NOREL_LABELS
    }
    all_pairs = set(itertools.combinations(mentions, 2))
    norel_pool = list(all_pairs - positive)
    n_norel = int(len(positive) * sampling_cfg.get("norel_ratio", 1.0))
    norel_sample = random.sample(norel_pool, min(n_norel, len(norel_pool)))
    selected = sorted(positive | set(norel_sample))
    return "\n".join(
        f'{s} ("{mentions_map.get(s, "")}"), {t} ("{mentions_map.get(t, "")}")'
        for s, t in selected
    )


def format_pair_lines(doc: dict, active_dataset: str, sampling_cfg: Optional[dict] = None) -> str:
    """Returns the pair_lines string for a document, matching the pipeline's prompt format."""
    if active_dataset in _PAIR_LIST_DATASETS:
        mentions_map = doc.get("mentions_map", {})
        # causal_timebank is binary-undirected: collapse to unordered pairs.
        undirected = (active_dataset == "causal_timebank")
        seen: set = set()
        lines = []
        for e1, e2 in doc.get("pair_list_ids", []):
            key = tuple(sorted([e1, e2])) if undirected else (e1, e2)
            if key in seen:
                continue
            seen.add(key)
            s, t = key
            lines.append(f'{s} ("{mentions_map.get(s, "")}"), {t} ("{mentions_map.get(t, "")}")')
        return "\n".join(lines)
    if active_dataset in _BINARY_SAMPLED_DATASETS:
        if sampling_cfg and sampling_cfg.get("enabled"):
            return _sample_pair_lines(doc, sampling_cfg)
        return "Predict all event mention pairs in the text; omit a pair to mark it NoRel."
    raise ValueError(f"format_pair_lines: unsupported dataset '{active_dataset}'")


def format_gold_output(gold_triples: List[Tuple[str, str, str]], active_dataset: str) -> str:
    """Formats gold triples as the JSON output the model is expected to produce."""
    if active_dataset in _PAIR_LIST_DATASETS:
        items = [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]
    elif active_dataset in _BINARY_SAMPLED_DATASETS:
        items = [
            {"pair": f"{src},{tgt}", "label": lbl}
            for src, lbl, tgt in gold_triples
            if lbl.lower() not in _NOREL_LABELS
        ]
    else:
        raise ValueError(f"format_gold_output: unsupported dataset '{active_dataset}'")
    return json.dumps(items)
