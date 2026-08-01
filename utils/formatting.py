from __future__ import annotations

import json
from typing import List, Optional, Tuple

from utils.labels import NOREL_VARIANTS, BINARY_POS, NOREL


# ── JSON answer parsing ───────────────────────────────────────────────────────
# Shared by main.py (parsing inference output) and tools/few_shot.py (checking a
# synthesized CoT still ends in the gold answer). Lives here rather than in main.py
# because main.py imports tools.few_shot, so few_shot cannot import back from it.

def _parsed_array_spans(text: str) -> List[Tuple[int, int, list]]:
    """Every top-level JSON array in ``text``, as ``(start, end, value)``, in order.

    Each candidate ``[`` is handed to the JSON decoder, which is the only thing that
    actually knows where a JSON value ends: it tracks strings and escapes correctly and
    rejects prose like ``[see below]`` on its own. A hand-rolled bracket scanner cannot —
    the previous one tracked quotes itself, so a single unbalanced ``"`` anywhere in the
    model's reasoning (quoting a text fragment, an inches mark) latched it into
    "inside a string" and it silently swallowed every array after that point, including
    the final answer.
    """
    decoder = json.JSONDecoder()
    out = []
    i = text.find("[")
    while i != -1:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("[", i + 1)
            continue
        out.append((i, end, value))
        i = text.find("[", end)  # skip nested arrays; only top-level ones are answers
    return out


def iter_json_array_spans(text: str):
    """Yield ``(start, end)`` slice bounds for every top-level JSON array in ``text``."""
    for start, end, _ in _parsed_array_spans(text):
        yield start, end


def iter_json_arrays(text: str):
    """Yield every top-level JSON array in ``text`` verbatim, in order."""
    for start, end, _ in _parsed_array_spans(text):
        yield text[start:end]


def _inference_answer_span(text: str) -> Optional[Tuple[int, int]]:
    """Slice bounds of the array to read as the answer of a *model under evaluation*.

    The answer is the final *populated* array; a stray ``[]`` in the reasoning must not
    win over it. An all-empty response falls back to the last ``[]``, which is a
    legitimate "no relations" result.
    """
    empty_span = None
    populated_span = None
    for start, end, data in _parsed_array_spans(text):
        if data:
            populated_span = (start, end)
        else:
            empty_span = (start, end)
    return populated_span or empty_span


def _literal_answer_span(text: str) -> Optional[Tuple[int, int]]:
    """Slice bounds of the *last* array in ``text``, populated or not.

    Used when the answer's location matters and not just its value — checking that a
    synthesized CoT ends in the gold labels, and splicing gold over what it does end in.
    _inference_answer_span's populated-wins rule is wrong for that: a response that
    quotes the prompt's illustrative array and then correctly concludes ``[]`` would be
    read as answering with the illustration, and a splice would overwrite the quote
    rather than the answer.
    """
    spans = _parsed_array_spans(text)
    return (spans[-1][0], spans[-1][1]) if spans else None


def final_json_array(text: str) -> Optional[str]:
    """The last JSON array in ``text`` verbatim, or None if there is none."""
    span = _literal_answer_span(text)
    return text[span[0]:span[1]] if span else None


def replace_final_json_array(text: str, replacement: str) -> str:
    """Return ``text`` with its last JSON array swapped for ``replacement``.

    When ``text`` contains no parseable array at all there is nothing to splice over,
    so the replacement is appended as the response's answer instead.
    """
    span = _literal_answer_span(text)
    if span is None:
        return f"{text.rstrip()}\n\n{replacement}".lstrip()
    return text[:span[0]] + replacement + text[span[1]:]


def _triples_from_array(data: list) -> List[Tuple[str, str, str]]:
    triples = []
    for item in data:
        pair = item.get("pair", "")
        if "," in pair:
            src, tgt = [part.strip() for part in pair.split(",", 1)]
            triples.append((src, item.get("label", "Unknown"), tgt))
    return triples


def parse_triples_from_text(content: str) -> List[Tuple[str, str, str]]:
    """Parse an LLM response into a list of ``(src, label, tgt)`` triples.

    Robust to conversational filler and reasoning. An answer of ``[]`` (no causal
    relations) parses to an empty list instead of raising. Raises ValueError when no
    JSON array is present anywhere in the response.
    """
    span = _inference_answer_span(content)
    if span is None:
        raise ValueError("No JSON array found in output")
    return _triples_from_array(json.loads(content[span[0]:span[1]]))


def parse_final_answer_triples(content: str) -> List[Tuple[str, str, str]]:
    """Like parse_triples_from_text, but reads the *last* array rather than the last
    populated one — see _literal_answer_span."""
    span = _literal_answer_span(content)
    if span is None:
        raise ValueError("No JSON array found in output")
    return _triples_from_array(json.loads(content[span[0]:span[1]]))


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
