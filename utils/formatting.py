from __future__ import annotations

import json
from typing import List, Tuple

_NOREL_LABELS = frozenset({"norel", "none", "no_rel"})


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


def format_pair_lines(doc: dict, active_dataset: str) -> str:
    """Returns the pair_lines string for a document, matching the pipeline's prompt format."""
    if active_dataset == "meci":
        mentions_map = doc.get("mentions_map", {})
        lines = [
            f'{src} ("{mentions_map.get(src, "")}"), {tgt} ("{mentions_map.get(tgt, "")}")'
            for src, _lbl, tgt in doc["gold_triples"]
        ]
        return "\n".join(lines)
    if active_dataset == "event_story_line":
        return "Predict all the pairs, all pairs not predicted will be considered NoRel"
    raise ValueError(f"format_pair_lines: unsupported dataset '{active_dataset}'")


def format_gold_output(gold_triples: List[Tuple[str, str, str]], active_dataset: str) -> str:
    """Formats gold triples as the JSON output the model is expected to produce."""
    if active_dataset == "meci":
        items = [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]
    elif active_dataset == "event_story_line":
        items = [
            {"pair": f"{src},{tgt}", "label": lbl}
            for src, lbl, tgt in gold_triples
            if lbl.lower() not in ("norel", "none")
        ]
    else:
        raise ValueError(f"format_gold_output: unsupported dataset '{active_dataset}'")
    return json.dumps(items)
