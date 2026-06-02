#!/usr/bin/env python
"""
preprocess_ann.py  ─────────────────────────────────────────────────────────────
Parse *.ann.tsvx event-relation files organised as
   <root>/<language>/{train,test,dev}/*.ann.tsvx

For every document we return
    • id         : file name without extension
    • lang       : language code (folder name)
    • split      : "train" | "test" | "dev"
    • tokens     : list[str]
    • mentions   : list[event_id]
    • spans      : list[list[int]]         (token indices)
    • relations  : dict[str, list[[src, tgt]]]

CLI usage
---------
python preprocess_ann.py \
    --root_dir data/ann_dataset \
    --repo_id  my-user/aviation-mentions \
    --exclude  aviation_accidents-week4-nhung-7946493_chunk_12.ann.tsvx \
    --shuffle 42 \
    --private
"""

from __future__ import annotations
import os, glob, argparse
from collections import defaultdict
from typing import Dict, List, Set
import re

import datasets
from datasets import Dataset, DatasetDict

# ──────────────────────────────────────────────────────────────────────────────
def _parse_ann_file(path: str) -> Dict:
    """Return one row (dict) extracted from a single *.ann.tsvx file."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    if not lines or not lines[0].startswith("Text\t"):
        raise ValueError(f"Malformed file (no Text line): {path}")

    # ── 1. tokenise text ────────────────────────────────────────────────────
    raw_text = lines[0].split("\t", 1)[1]

    # Use regex to grab every non-whitespace span:
    token_matches = list(re.finditer(r"\S+", raw_text))
    tokens: List[str]  = [m.group(0) for m in token_matches]
    starts: List[int]  = [m.start()   for m in token_matches]   # char → token idx

    # Helper: char-offset  ➜  token-index
    def char_to_tok(char_idx: int) -> int:
        # Linear scan is fine (docs are short), but bisect
        # would be O(log n) if you prefer.
        for i, s in enumerate(starts):
            if s <= char_idx < s + len(tokens[i]):
                return i
        raise ValueError(f"Offset {char_idx} outside text in {path}")

    # ── 2. mentions & spans ───────────────────────────────────────────────────
    mentions, spans = [], []
    for ln in lines[1:]:
        if not ln.startswith("Event\t"):
            continue
        # Format:  Event  T0  surviving  EVENT  11
        _, e_id, surface, _label, char_off = ln.split("\t")[:5]
        off = int(char_off)

        tok_idx = char_to_tok(off)

        # If the surface form contains spaces, the mention is multi-token.
        length = len(surface.split())
        spans.append(list(range(tok_idx, tok_idx + length)))
        mentions.append(e_id)

    # 3. relations -------------------------------------------------------------
    relations: Dict[str, List[List[str]]] = defaultdict(list)
    for ln in lines[1:]:
        if not ln.startswith("Relation\t"):
            continue
        _, src, tgt, rel_type, *_ = ln.split("\t")
        src_idx = mentions.index(src)
        tgt_idx = mentions.index(tgt)
        relations[rel_type].append([src_idx, tgt_idx])

    # Rename relation types to canonical labels
    _REL_RENAME = {"CauseEffect": "causes", "EffectCause": "causedby", "NoRel": "norel"}
    relations = {_REL_RENAME.get(k, k): v for k, v in relations.items()}

    seen_pairs: set = set()
    pair_list: List[List[int]] = []
    for pairs in relations.values():
        for src_idx, tgt_idx in pairs:
            if (src_idx, tgt_idx) not in seen_pairs:
                seen_pairs.add((src_idx, tgt_idx))
                pair_list.append([src_idx, tgt_idx])

    return {
        "id":        os.path.splitext(os.path.basename(path))[0],
        "tokens":    tokens,
        "mentions":  mentions,
        "spans":     spans,
        "relations": dict(relations),
        "pair_list": pair_list,
    }

# ──────────────────────────────────────────────────────────────────────────────
def build_dataset_dict(root_dir: str, exclude: Set[str]) -> DatasetDict:
    """Walk every <language>/<split> directory and build a DatasetDict."""
    split_rows = defaultdict(list)          # split → list[dict]

    for lang_dir in glob.glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(lang_dir):
            continue
        lang = os.path.basename(lang_dir)

        for split in ("train", "test", "dev"):
            pattern = os.path.join(lang_dir, split, "*.ann.tsvx")
            for path in glob.glob(pattern):
                if os.path.basename(path) in exclude:
                    print(f"🚫  Skipping excluded file: {path}")
                    continue
                row = _parse_ann_file(path)
                row["lang"]  = lang
                row["split"] = split
                split_rows[split].append(row)

    if not split_rows:
        raise RuntimeError(f"No documents found under {root_dir}")

    # Convert to HuggingFace datasets
    return DatasetDict({sp: Dataset.from_list(rows) for sp, rows in split_rows.items()})

# ──────────────────────────────────────────────────────────────────────────────
def push_to_hub(dd: DatasetDict, repo_id: str, private: bool, token: str | None):
    dd.push_to_hub(repo_id,
                   token=token or os.getenv("HF_TOKEN"),
                   private=private)
    print(f"✅  Dataset pushed to https://huggingface.co/datasets/{repo_id}")

# ─────────────────────────── CLI ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Parse SML XML files and push a HuggingFace dataset."
    )
    p.add_argument("--root_dir", default="data/MECI/meci-v0.1-public",
                   help="Root directory containing XML files.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated list of filenames (or doc_name) to skip.")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed (omit to skip shuffling).")
    p.add_argument("--test_size", type=float, default=0.1,
                   help="Train/test split fraction, e.g. 0.1 (omit for no split).")
    p.add_argument("--private", action="store_true",
                   help="If set, push the dataset privately.")
    p.add_argument("--token", default=None,
                   help="HF token (defaults to $HF_TOKEN env-var).")
    return p.parse_args()

# ─────────────────────────── main ──────────────────────────────────────────────
def main():
    args = parse_args()
    
    exclude_set = {e.strip() for e in args.exclude.split(",") if e.strip()}
    
    ds_dict = build_dataset_dict("data/MECI/meci-v0.1-public", exclude_set)
    
    # Optional shuffle for reproducible training
    
    for sp in ds_dict:
        ds_dict[sp] = ds_dict[sp].shuffle(seed=42)
    
    push_to_hub(ds_dict,
                repo_id="Nofing/MECI-v0.1-public-standard-eci",
                private=False,
                token=None)


if __name__ == "__main__":
    main()