#!/usr/bin/env python3
"""
CausalTimeBank — standard ECI dataprep
=======================================
Aligned with the standard_eci schema used by EventStoryLine_dataprep_standard_eci.py.

Decisions
---------
mentions      eiids (event instance IDs), position-ordered
relations     intra-sentence CLINKs only (286 / 318 = 90%); cross-sentence dropped.
              Each CLINK a→b produces:
                "causes":   [a_idx, b_idx]
                "causedby": [b_idx, a_idx]
              NoRel not stored — only positive pairs.
pair_list     all directed intra-sentence mention pairs; used for prompt constraint
              and metric filtering (constrain_to_pair_list).
split         no official split — all 183 docs in "train"; 5-fold CV via doc_idx % 5.

Output schema per row
---------------------
  id         str
  doc_idx    int                stable sequential document index
  tokens     list[str]          flat token list (all sentences)
  mentions   list[str]          eiids, position-ordered
  spans      list[list[int]]    token index lists per mention (0-based, doc-level)
  relations  dict               {"causes":   [[src_idx, tgt_idx], ...],
                                 "causedby": [[tgt_idx, src_idx], ...]}
  sentences  list[[int, int]]   sentence token boundary pairs [start, end)
  pair_list  list[[int, int]]   all directed intra-sentence pairs (mention indices)

Fold usage (consumer-side)
--------------------------
  train = ds["train"].filter(lambda x: x["doc_idx"] % 5 != k)
  test  = ds["train"].filter(lambda x: x["doc_idx"] % 5 == k)

References
----------
  Mirza et al. (COLING 2014)         https://aclanthology.org/C14-1198
  Mirza & Tonelli (CATENA, COLING 2016)  https://aclanthology.org/C16-1007
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value


# ─────────────────── sentence-aware TEXT element parsing ─────────────────────

def _extract_tokens_and_sentences(
    text_el: ET.Element,
) -> Tuple[List[str], Dict[str, List[int]], List[List[int]]]:
    tokens: List[str] = []
    eid_to_span: Dict[str, List[int]] = {}
    sentence_starts: List[int] = [0]

    def _add(text: str | None, eid: str | None = None) -> None:
        if not text:
            return
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0 and tokens:
                sentence_starts.append(len(tokens))
            words = line.split()
            if not words:
                continue
            if eid is not None:
                start = len(tokens)
                tokens.extend(words)
                eid_to_span[eid] = list(range(start, len(tokens)))
            else:
                tokens.extend(words)

    _add(text_el.text)
    for child in text_el:
        child_eid = child.get("eid") if child.tag == "EVENT" else None
        _add(child.text, eid=child_eid)
        _add(child.tail)

    sentence_ends = sentence_starts[1:] + [len(tokens)]
    sentences = [[s, e] for s, e in zip(sentence_starts, sentence_ends) if s < e]
    return tokens, eid_to_span, sentences


# ──────────────────────────────── .tml parser ────────────────────────────────

def _parse_tml(path: str, doc_idx: int) -> Dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        print(f"  Skipping malformed XML {path}: {exc}")
        return None
    root = tree.getroot()

    docid_el = root.find("DOCID")
    doc_id = (
        docid_el.text.strip()
        if docid_el is not None
        else os.path.splitext(os.path.basename(path))[0]
    )

    text_el = root.find("TEXT")
    if text_el is None:
        print(f"  No <TEXT> in {path}, skipping")
        return None

    tokens, eid_to_span, sentences = _extract_tokens_and_sentences(text_el)

    eiid_to_eid: Dict[str, str] = {}
    for mi in root.findall("MAKEINSTANCE"):
        eiid = mi.get("eiid")
        eid  = mi.get("eventID")
        if eiid and eid:
            eiid_to_eid[eiid] = eid

    eiid_span_pairs = [
        (eiid, eid_to_span[eid])
        for eiid, eid in eiid_to_eid.items()
        if eid in eid_to_span
    ]
    eiid_span_pairs.sort(key=lambda x: x[1][0])

    mentions     = [eiid for eiid, _ in eiid_span_pairs]
    spans        = [span for _, span in eiid_span_pairs]
    eiid_to_idx  = {eiid: i for i, eiid in enumerate(mentions)}

    midx_to_sent: Dict[int, int] = {}
    for sent_idx, (sent_start, sent_end) in enumerate(sentences):
        for m_idx, span in enumerate(spans):
            if span and sent_start <= span[0] < sent_end:
                midx_to_sent[m_idx] = sent_idx

    causes:   List[List[int]] = []
    causedby: List[List[int]] = []
    for clink in root.findall("CLINK"):
        src = clink.get("eventInstanceID")
        tgt = clink.get("relatedToEventInstance")
        if not (src and tgt and src in eiid_to_idx and tgt in eiid_to_idx):
            continue
        src_idx, tgt_idx = eiid_to_idx[src], eiid_to_idx[tgt]
        if midx_to_sent.get(src_idx) == midx_to_sent.get(tgt_idx):
            causes.append([src_idx, tgt_idx])
            causedby.append([tgt_idx, src_idx])

    sent_to_midxs: Dict[int, List[int]] = defaultdict(list)
    for m_idx, sent_idx in midx_to_sent.items():
        sent_to_midxs[sent_idx].append(m_idx)

    pair_list: List[List[int]] = []
    for midxs in sent_to_midxs.values():
        if len(midxs) < 2:
            continue
        midxs_sorted = sorted(midxs, key=lambda i: spans[i][0])
        for i in midxs_sorted:
            for j in midxs_sorted:
                if i != j:
                    pair_list.append([i, j])

    return {
        "id":        doc_id,
        "doc_idx":   doc_idx,
        "tokens":    tokens,
        "mentions":  mentions,
        "spans":     spans,
        "relations": {"causes": causes, "causedby": causedby},
        "sentences": sentences,
        "pair_list": pair_list,
    }


# ──────────────────────────── dataset builder ────────────────────────────────

def build_dataset(root_dir: str) -> DatasetDict:
    files = sorted(glob.glob(os.path.join(root_dir, "**", "*.tml"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .tml files found under {root_dir}")

    rows = [_parse_tml(f, idx) for idx, f in enumerate(files)]
    rows = [r for r in rows if r is not None]

    n_causes  = sum(len(r["relations"]["causes"]) for r in rows)
    n_pairs   = sum(len(r["pair_list"]) for r in rows)
    print(f"Documents              : {len(rows)} / {len(files)}")
    print(f"Intra-sentence pairs   : {n_pairs}")
    print(f"causes                 : {n_causes}  →  {n_causes * 2} directional")

    features = Features({
        "id":        Value("string"),
        "doc_idx":   Value("int64"),
        "tokens":    Sequence(Value("string")),
        "mentions":  Sequence(Value("string")),
        "spans":     Sequence(Sequence(Value("int64"))),
        "relations": {
            "causes":   Sequence(Sequence(Value("int64"))),
            "causedby": Sequence(Sequence(Value("int64"))),
        },
        "sentences": Sequence(Sequence(Value("int64"))),
        "pair_list": Sequence(Sequence(Value("int64"))),
    })
    return DatasetDict({
        "train": Dataset.from_list(rows, features=features),
    })


# ──────────────────────────────── HF push ────────────────────────────────────

def push_to_hub(dataset: DatasetDict, repo_id: str, token: str | None, private: bool) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────── CLI ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CausalTimeBank → standard ECI HF dataset (intra-sentence CLINKs only)."
    )
    p.add_argument(
        "--root_dir",
        default="data/CausalTimeBank/TimeML",
        help="Directory containing .tml files (after unzipping Causal-TimeBank-TimeML.zip).",
    )
    p.add_argument("--repo_id", default="Nofing/CausalTimeBank-standard-eci")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (defaults to $HF_TOKEN).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(root_dir=args.root_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
