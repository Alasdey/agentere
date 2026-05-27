from __future__ import annotations

import json
from typing import List, Optional, Tuple

from utils.labels import NOREL_VARIANTS, BINARY_POS, NOREL


def convert_to_binary_undirected(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Collapse causes/causedby → BINARY_POS, deduplicate by unordered pair."""
    seen: set = set()
    result = []
    for src, lbl, tgt in triples:
        key = frozenset([src, tgt])
        if key in seen:
            continue
        seen.add(key)
        canonical = tuple(sorted([src, tgt]))
        label = NOREL if lbl.lower() in NOREL_VARIANTS else BINARY_POS
        result.append((canonical[0], label, canonical[1]))
    return result


def format_pair_lines(
    doc: dict,
    binary_undirected: bool = False,
    constrain_to_pair_list: bool = True,
) -> str:
    """
    Returns the pair_lines string injected into the prompt.

    Dispatch is driven by pair_list_ids presence + constrain_to_pair_list:
    - pair_list_ids present AND constrain=True  → enumerate explicit pairs (MECI/CTB).
      binary_undirected collapses (A,B)/(B,A) to one canonical entry.
    - Otherwise → open-ended instruction (ESL/Maven-ERE, or MECI/CTB unconstrained).

    constrain_to_pair_list comes from config["datasets"][active_ds]["constrain_to_pair_list"].
    It is a no-op when pair_list_ids is absent from the doc.
    """
    pair_list_ids = doc.get("pair_list_ids")
    if pair_list_ids and constrain_to_pair_list:
        mentions_map = doc.get("mentions_map", {})
        seen: set = set()
        lines = []
        for e1, e2 in pair_list_ids:
            key = tuple(sorted([e1, e2])) if binary_undirected else (e1, e2)
            if key in seen:
                continue
            seen.add(key)
            s, t = key
            lines.append(f'{s} ("{mentions_map.get(s, "")}"), {t} ("{mentions_map.get(t, "")}")')
        return "\n".join(lines)

    # Open-ended: model enumerates candidates; omission signals norel.
    return "Predict all event mention pairs in the text; omit a pair to mark it norel."


def format_gold_output(
    gold_triples: List[Tuple[str, str, str]],
    pair_list_ids: Optional[list] = None,
) -> str:
    """
    Formats gold triples as the JSON output the model is expected to produce.

    pair_list_ids present (MECI, CTB): gold includes explicit norel entries —
        output every triple so the model sees a label for every listed pair.
    pair_list_ids absent (ESL, Maven-ERE): model only outputs positive pairs;
        omission encodes norel — filter norel from gold.
    """
    if pair_list_ids:
        items = [{"pair": f"{src},{tgt}", "label": lbl} for src, lbl, tgt in gold_triples]
    else:
        items = [
            {"pair": f"{src},{tgt}", "label": lbl}
            for src, lbl, tgt in gold_triples
            if lbl.lower() not in NOREL_VARIANTS
        ]
    return json.dumps(items)
