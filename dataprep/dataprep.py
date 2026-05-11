
from __future__ import annotations

import re
from typing import Set, List, Dict, Any, Optional, Iterator, Tuple
from datasets import load_dataset

# =============================================================================
# REGEX CONSTANTS
# =============================================================================

# Matches a relation triple in the annotation string.
# Pattern: <ID CONTENT> LABEL <ID CONTENT>
# Constraints:
#   1. ID cannot contain whitespace or '>'.
#   2. CONTENT cannot contain newlines.
# We capture:
#   Group 1: Source ID
#   Group 3: Relation Label
#   Group 4: Target ID
# Groups 2 and 5 (Content) are matched to ensure validity but ignored in output.
RELATION_TRIPLE_RE = re.compile(
    r"<([^>\s]+)\s+([^>\n]+)>"       # Source Tag: <ID Content>
    r"\s+"                           # Space
    r"([A-Za-z0-9_\-]+)"             # Relation Label
    r"\s+"                           # Space
    r"<([^>\s]+)\s+([^>\n]+)>"       # Target Tag: <ID Content>
)


def parse_annotations(
    ann_text: str, 
    valid_labels: Optional[Set[str]] = None
) -> List[Tuple[str, str, str]]:
    """
    Parses an annotation string containing relations into triples.
    Labels and IDs are extracted exactly as they appear (no normalization).
    
    Args:
        ann_text: The raw annotation string.
        valid_labels: If provided, only keep relations strictly matching these strings.

    Returns:
        List[Tuple[str, str, str]]: A list of (source_id, label, target_id) triples.
    """
    triples: List[Tuple[str, str, str]] = []
    if not ann_text:
        return triples

    for m in RELATION_TRIPLE_RE.finditer(ann_text):
        src_id = m.group(1)
        label = m.group(3)
        tgt_id = m.group(4)
        
        if valid_labels and label not in valid_labels:
            continue
            
        triples.append((src_id, label, tgt_id))

    return triples


def load_hf_dataset_parsed(
    repo_id: str,
    split: str = "test",
    text_field: str = "text",
    ann_field: str = "annots",
    valid_labels: Optional[Set[str]] = None,
    streaming: bool = False,
    max_examples: int = 0,
    binary_undirected: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Loads a Hugging Face dataset and yields the raw text and parsed relation triples.

    Args:
        repo_id: Hugging Face dataset path.
        split: Dataset split.
        text_field: Column name containing the document text.
        ann_field: Column name containing the relation strings.
        valid_labels: Set of exact label strings to keep.
        streaming: Whether to stream the dataset.
        max_examples: Max number of examples to yield (0 = all).

    Yields:
        Dict containing:
        - "id": Row ID
        - "doc_text": Raw document text (untouched)
        - "gold_triples": List of (src_id, label, tgt_id)
        - "lang": Language code
    """
    
    ds = load_dataset(repo_id, split=split, streaming=streaming)
    iterable = ds if streaming else (ds[i] for i in range(len(ds)))

    count = 0
    for i, row in enumerate(iterable):
        if max_examples > 0 and count >= max_examples:
            break
            
        doc_text = row.get(text_field, "")
        ann_text = row.get(ann_field, "")
        
        # Use existing 'id' column or fallback to index
        row_id = str(row.get("id", f"idx_{i}"))
        lang = row.get("lang", "eng")
        tokens = row.get("tokens", [])
        spans = row.get("spans", [])
        mentions = row.get("mentions", [])

        gold_triples = parse_annotations(ann_text, valid_labels=None)
        if binary_undirected:
            from utils.formatting import convert_to_binary_undirected
            gold_triples = convert_to_binary_undirected(gold_triples)
        elif valid_labels:
            gold_triples = [(s, l, t) for s, l, t in gold_triples if l in valid_labels]
        
        mentions_map = {}
        if tokens and spans and mentions and len(spans) == len(mentions):
            for j, mention_id in enumerate(mentions):
                token_indices = spans[j]
                # Ensure indices are valid and gather the text parts
                text_parts = []
                for idx in token_indices:
                    if 0 <= idx < len(tokens):
                        text_parts.append(tokens[idx])
                
                if text_parts:
                    mentions_map[mention_id] = " ".join(text_parts)

        yield {
            "id": row_id,
            "doc_idx": i,
            "doc_text": doc_text,
            "gold_triples": gold_triples,
            "lang": lang,
            "mentions_map": mentions_map,
        }
        
        count += 1