from __future__ import annotations

import json
from typing import List, Tuple


def format_pair_lines(doc: dict, active_dataset: str) -> str:
    """Returns the pair_lines string for a document, matching the pipeline's prompt format."""
    if active_dataset in ("meci", "maven_ere"):
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
    if active_dataset in ("meci", "maven_ere"):
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
