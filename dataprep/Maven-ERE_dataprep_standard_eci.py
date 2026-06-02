#!/usr/bin/env python3
"""
MAVEN-ERE — standard ECI dataprep
==================================
Aligned with the standard_eci schema used by EventStoryLine_dataprep_standard_eci.py
and CausalTimeBank_dataprep_standard_eci.py.

Decisions
---------
mentions      Events only (no TIMEX); position-ordered as e0…eN.
relations     CAUSE + PRECONDITION both map to causes/causedby (directional pair).
              Coreference is expanded: concept-level annotations are expanded to all
              mention-level pairs via the events[].mention lists.
              NoRel not stored — only positive pairs.
split         train.jsonl → 90% train / 10% dev (random seed=42).
              valid.jsonl → test.
              No pair_list (all mention pairs are candidates).

Output schema per row
---------------------
  id         str
  doc_idx    int                stable sequential document index
  tokens     list[str]          flat token list (all sentences)
  mentions   list[str]          e{N} IDs, position-ordered (sent_id, offset[0])
  spans      list[list[int]]    token index lists per mention (0-based, doc-level)
  relations  dict               {"causes":   [[src_idx, tgt_idx], ...],
                                 "causedby": [[tgt_idx, src_idx], ...]}
  sentences  list[[int, int]]   sentence token boundary pairs [start, end)

References
----------
  Wang et al. (EMNLP 2022)  https://aclanthology.org/2022.emnlp-main.60
"""
from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value


# ──────────────────────────────── doc parser ─────────────────────────────────

def _parse_doc(raw: dict, doc_idx: int) -> Dict | None:
    doc_id = raw.get("id", f"doc_{doc_idx}")
    words_nested: List[List[str]] = raw.get("tokens", [])

    # Flatten tokens and compute sentence boundaries [start, end)
    tokens: List[str] = []
    sent_starts: List[int] = []
    for sent in words_nested:
        sent_starts.append(len(tokens))
        tokens.extend(sent)
    if not tokens:
        return None
    sent_ends = sent_starts[1:] + [len(tokens)]
    sentences = [[s, e] for s, e in zip(sent_starts, sent_ends) if s < e]

    # Collect all event mentions (events only, no TIMEX), sorted by position
    raw_mentions = []
    for concept in raw.get("events", []):
        concept_id = concept["id"]
        for mention in concept.get("mention", []):
            raw_mentions.append({
                "mention_id":  mention["id"],
                "concept_id":  concept_id,
                "sent_id":     mention["sent_id"],
                "offset":      mention["offset"],  # [start, end) within sentence
            })

    raw_mentions.sort(key=lambda m: (m["sent_id"], m["offset"][0]))

    # Build e{N} IDs and doc-level spans
    mentions: List[str] = []
    spans: List[List[int]] = []
    concept_to_enums: Dict[str, List[str]] = defaultdict(list)
    enum_to_idx: Dict[str, int] = {}

    for i, m in enumerate(raw_mentions):
        enum = f"e{i}"
        mentions.append(enum)
        sent_off = sent_starts[m["sent_id"]]
        span = list(range(sent_off + m["offset"][0], sent_off + m["offset"][1]))
        spans.append(span)
        concept_to_enums[m["concept_id"]].append(enum)
        enum_to_idx[enum] = i

    # Extract CAUSE + PRECONDITION, expand through concept-level coreference
    causes: List[List[int]] = []
    causedby: List[List[int]] = []
    seen: set = set()

    causal_rels = raw.get("causal_relations", {})
    for rel_type in ("CAUSE", "PRECONDITION"):
        for pair in causal_rels.get(rel_type, []):
            src_concept, tgt_concept = pair[0], pair[1]
            src_enums = concept_to_enums.get(src_concept, [])
            tgt_enums = concept_to_enums.get(tgt_concept, [])
            for src_enum in src_enums:
                for tgt_enum in tgt_enums:
                    key = (src_enum, tgt_enum)
                    if key not in seen:
                        seen.add(key)
                        src_idx = enum_to_idx[src_enum]
                        tgt_idx = enum_to_idx[tgt_enum]
                        causes.append([src_idx, tgt_idx])
                        causedby.append([tgt_idx, src_idx])

    return {
        "id":        doc_id,
        "doc_idx":   doc_idx,
        "tokens":    tokens,
        "mentions":  mentions,
        "spans":     spans,
        "relations": {"causes": causes, "causedby": causedby},
        "sentences": sentences,
    }


# ──────────────────────────────── JSONL loader ───────────────────────────────

def _load_jsonl(path: str, doc_idx_offset: int) -> Tuple[List[Dict], int]:
    rows = []
    doc_idx = doc_idx_offset
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            row = _parse_doc(raw, doc_idx)
            if row is not None:
                rows.append(row)
                doc_idx += 1
    return rows, doc_idx


# ──────────────────────────────── dataset builder ────────────────────────────

def build_dataset(data_dir: str) -> DatasetDict:
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")

    train_rows, offset = _load_jsonl(train_path, doc_idx_offset=0)
    test_rows, _       = _load_jsonl(valid_path, doc_idx_offset=offset)

    n_causes_train = sum(len(r["relations"]["causes"]) for r in train_rows)
    n_causes_test  = sum(len(r["relations"]["causes"]) for r in test_rows)
    print(f"Train docs             : {len(train_rows)}")
    print(f"Test  docs (valid.jsonl): {len(test_rows)}")
    print(f"causes (train)         : {n_causes_train}  →  {n_causes_train * 2} directional")
    print(f"causes (test)          : {n_causes_test}  →  {n_causes_test * 2} directional")

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
    })

    full_train = Dataset.from_list(train_rows, features=features)
    split = full_train.train_test_split(test_size=0.1, seed=42)

    return DatasetDict({
        "train": split["train"],
        "dev":   split["test"],
        "test":  Dataset.from_list(test_rows, features=features),
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
        description="MAVEN-ERE → standard ECI HF dataset (CAUSE+PRECONDITION, events only)."
    )
    p.add_argument(
        "--data_dir",
        default="data/MAVEN-ERE/MAVEN_ERE/",
        help="Directory containing train.jsonl and valid.jsonl.",
    )
    p.add_argument("--repo_id", default="Nofing/MAVEN-ERE-standard-eci")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (defaults to $HF_TOKEN).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(data_dir=args.data_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
