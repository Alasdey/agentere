from __future__ import annotations
from typing import List, Tuple

BINARY_LABELS = ["Rel", "NoRel"]


def to_binary_label(label: str) -> str:
    return "NoRel" if label.lower() in ("norel", "none", "unknown") else "Rel"


def collapse_to_binary(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Collapse multi-class triples to Rel/NoRel and deduplicate undirected pairs.

    (A,B) and (B,A) are treated as the same pair — Rel wins over NoRel.
    Canonical order is alphabetical sort of the two IDs.
    """
    pair_labels: dict[tuple, str] = {}
    for src, lbl, tgt in triples:
        canonical = tuple(sorted([src, tgt]))
        binary = to_binary_label(lbl)
        if canonical not in pair_labels or binary == "Rel":
            pair_labels[canonical] = binary
    return [(src, lbl, tgt) for (src, tgt), lbl in pair_labels.items()]
