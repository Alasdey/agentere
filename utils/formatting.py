from __future__ import annotations

import json
from typing import List, Tuple


def format_pair_lines(doc: dict, active_dataset: str) -> str:
    """Returns the pair_lines string for a document, matching the pipeline's prompt format."""
    if active_dataset == "meci":
        mentions_map = doc.get("mentions_map", {})
        lines = [
            f'{src} ("{mentions_map.get(src, "")}"), {tgt} ("{mentions_map.get(tgt, "")}")'
            for src, _lbl, tgt in doc["gold_triples"]
        ]
        return "\n".join(lines)
    return "Predict all the pairs, all pairs not predicted will be considered NoRel"


def format_gold_output(gold_triples: List[Tuple[str, str, str]], active_dataset: str) -> str:
    """Formats gold triples as the JSON output the model is expected to produce."""
    if active_dataset == "meci":
        # Closed-set: include all pairs with their labels
        items = [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]
    else:
        # Open extraction: only positive (non-NoRel) triples
        items = [
            {"pair": f"{src},{tgt}", "label": lbl}
            for src, lbl, tgt in gold_triples
            if lbl.lower() not in ("norel", "none")
        ]
    return json.dumps(items)
