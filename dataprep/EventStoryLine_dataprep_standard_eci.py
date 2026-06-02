#!/usr/bin/env python3
"""
EventStoryLine v0.9 — ECI dataprep (master)
============================================
Schema aligned with Nofing/CausalTimeBank-standard.

Decisions
---------
version       v0.9  — the only version benchmarked by published ECI papers.
mentions      ACTION_* tags only (event triggers).
relations     Loaded from evaluation_format/full_corpus/v0.9/event_mentions_extended/
              (pre-computed by create_gold_document.py, coreference-propagated via
              ECB+ chains → 5,625 pairs across 258 docs).
              relType ∈ {PRECONDITION, FALLING_ACTION} → causes [src, tgt] /
              causedby [tgt, src].
              5 docs have no tab file and fall back to raw XML PLOT_LINKs.
dev topics    {37, 41} — Caselli & Vossen (2017), confirmed in HOTECI source.
              split="dev", excluded from 5-fold CV.
fold          doc_idx % 5  (train docs only).
doc_idx       contiguous — assigned only to rows that parse successfully.
coref         Propagated via evaluation format (standard across all SOTA papers).
NOTE          pair_list not stored — consumers enumerate ordered mention pairs at
              training time using sentences + spans.

Output schema (one row per document)
-------------------------------------
  id          str
  doc_idx     int          stable 0-based index, sorted by (topic_id, filename)
  topic_id    int          numeric topic folder (1, 3, … 41)
  split       str          "train" | "dev"
  tokens      list[str]    flat token list (doc-level)
  mentions    list[str]    m_id strings, sorted by first token position
  spans       list[list[int]]  token indices per mention (0-based, doc-level)
  relations   dict         {"causes":    [[src_idx, tgt_idx], …],
                             "causedby": [[tgt_idx, src_idx], …]}
  sentences   list[list[int]]  [[start, end), …] token boundary slices

Fold usage
----------
  train = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 != k)
  test  = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 == k)
  dev   = ds.filter(lambda x: x["split"] == "dev")

References
----------
  Caselli & Vossen, ACL 2017  https://aclanthology.org/W17-2711
  Man et al. (DiffusECI), AAAI 2024
  Liao et al. (CLECI), Information 2026
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value

# ── constants ─────────────────────────────────────────────────────────────────

DEV_TOPICS: set[int] = {37, 41}

ACTION_TAGS: set[str] = {
    "ACTION_OCCURRENCE", "ACTION_REPORTING", "ACTION_PERCEPTION",
    "ACTION_ASPECTUAL",  "ACTION_STATE",     "ACTION_CAUSATIVE",
    "ACTION_GENERIC",    "ACTION_GOLD",       "ACTION_SILVER",
    "NEG_ACTION_OCCURRENCE", "NEG_ACTION_REPORTING", "NEG_ACTION_PERCEPTION",
    "NEG_ACTION_ASPECTUAL",  "NEG_ACTION_STATE",     "NEG_ACTION_CAUSATIVE",
    "NEG_ACTION_GENERIC",
}

CAUSAL_RELTYPES: set[str] = {"PRECONDITION", "FALLING_ACTION"}

# ── XML parser ────────────────────────────────────────────────────────────────

def _parse_xml(path: str, doc_idx: int, topic_id: int, eval_dir: str) -> Optional[Dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"  Skipping malformed XML {path}: {exc}")
        return None

    doc_id: str = root.get("doc_name") or os.path.splitext(os.path.basename(path))[0]

    # 1. Tokens — each <token> carries a `sentence` integer attribute
    tokens: List[str] = []
    tid_to_pos: Dict[int, int] = {}
    pos_to_sent: Dict[int, int] = {}

    for tok in sorted(root.findall(".//token"), key=lambda t: int(t.get("t_id", 0))):
        t_id  = int(tok.get("t_id", 0))
        sent  = int(tok.get("sentence", 0))
        pos   = len(tokens)
        tokens.append(tok.text or "")
        tid_to_pos[t_id] = pos
        pos_to_sent[pos]  = sent

    # 2. Sentence boundaries [start, end)
    sent_buckets: Dict[int, List[int]] = defaultdict(list)
    for pos, sid in pos_to_sent.items():
        sent_buckets[sid].append(pos)

    sentences: List[List[int]] = [
        [sorted(v)[0], sorted(v)[-1] + 1]
        for _, v in sorted(sent_buckets.items())
    ]

    # 3. ACTION mentions, sorted by first token position
    #    Also collect frozenset of t_id strings per mention for tab-file lookup.
    raw: List[Tuple[int, str, List[int], frozenset]] = []
    for m in root.findall(".//Markables/*"):
        if m.tag not in ACTION_TAGS:
            continue
        m_id = m.get("m_id")
        if not m_id:
            continue
        anchor_pairs = [
            (a.get("t_id"), tid_to_pos[int(a.get("t_id"))])
            for a in m.findall("token_anchor")
            if a.get("t_id") and int(a.get("t_id", 0)) in tid_to_pos
        ]
        if anchor_pairs:
            raw.append((
                min(p[1] for p in anchor_pairs),
                m_id,
                sorted(p[1] for p in anchor_pairs),
                frozenset(p[0] for p in anchor_pairs),
            ))

    raw.sort(key=lambda x: x[0])
    mentions: List[str]        = [r[1] for r in raw]
    spans:    List[List[int]]  = [r[2] for r in raw]
    mid_to_idx: Dict[str, int] = {m: i for i, m in enumerate(mentions)}
    tid_set_to_midx: Dict[frozenset, int] = {r[3]: i for i, r in enumerate(raw)}

    if len(mentions) < 2:
        return None

    # 4. Causal relations — from propagated evaluation format tab file.
    #    Falls back to raw XML PLOT_LINKs for the 5 docs without a tab file.
    causes:     List[List[int]] = []
    causedby:  List[List[int]] = []

    doc_stem = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
    tab_path = os.path.join(eval_dir, str(topic_id), doc_stem + ".tab")

    if os.path.exists(tab_path):
        with open(tab_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                src_str, tgt_str, rel_type = parts
                if rel_type not in CAUSAL_RELTYPES:
                    continue
                src_idx = tid_set_to_midx.get(frozenset(src_str.split("_")))
                tgt_idx = tid_set_to_midx.get(frozenset(tgt_str.split("_")))
                if src_idx is None or tgt_idx is None:
                    continue
                causes.append([src_idx, tgt_idx])
                causedby.append([tgt_idx, src_idx])
    else:
        for rel in root.findall(".//Relations/PLOT_LINK"):
            if rel.get("relType", "") not in CAUSAL_RELTYPES:
                continue
            src_el = rel.find("source")
            tgt_el = rel.find("target")
            if src_el is None or tgt_el is None:
                continue
            src_idx = mid_to_idx.get(src_el.get("m_id", ""))
            tgt_idx = mid_to_idx.get(tgt_el.get("m_id", ""))
            if src_idx is None or tgt_idx is None:
                continue
            causes.append([src_idx, tgt_idx])
            causedby.append([tgt_idx, src_idx])

    return {
        "id":       doc_id,
        "doc_idx":  doc_idx,
        "topic_id": topic_id,
        "split":    "dev" if topic_id in DEV_TOPICS else "train",
        "tokens":   tokens,
        "mentions": mentions,
        "spans":    spans,
        "relations": {
            "causes":    causes,
            "causedby": causedby,
        },
        "sentences": sentences,
    }

# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset(root_dir: str, eval_dir: Optional[str] = None) -> DatasetDict:
    xml_files = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found under {root_dir}")

    def _key(p: str) -> Tuple[int, str]:
        t = os.path.basename(os.path.dirname(p))
        return (int(t) if t.isdigit() else 0, os.path.basename(p))

    xml_files = sorted(xml_files, key=_key)

    if eval_dir is None:
        esl_root = os.path.dirname(os.path.dirname(root_dir))
        eval_dir = os.path.join(
            esl_root, "evaluation_format", "full_corpus", "v0.9", "event_mentions_extended"
        )

    rows: List[Dict] = []
    doc_idx = 0
    for path in xml_files:
        t = os.path.basename(os.path.dirname(path))
        row = _parse_xml(path, doc_idx, int(t) if t.isdigit() else 0, eval_dir)
        if row is not None:
            rows.append(row)
            doc_idx += 1

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_dev   = sum(1 for r in rows if r["split"] == "dev")
    n_ce    = sum(len(r["relations"]["causes"]) for r in rows)

    print(f"Documents : {len(rows)}  ({n_train} train / {n_dev} dev)")
    print(f"causes    : {n_ce}  →  {n_ce * 2} directional")

    features = Features({
        "id":       Value("string"),
        "doc_idx":  Value("int64"),
        "topic_id": Value("int64"),
        "split":    Value("string"),
        "tokens":   Sequence(Value("string")),
        "mentions": Sequence(Value("string")),
        "spans":    Sequence(Sequence(Value("int64"))),
        "relations": {
            "causes":    Sequence(Sequence(Value("int64"))),
            "causedby": Sequence(Sequence(Value("int64"))),
        },
        "sentences": Sequence(Sequence(Value("int64"))),
    })
    full = Dataset.from_list(rows, features=features)
    return DatasetDict({
        "train": full.filter(lambda x: x["split"] == "train"),
        "dev":   full.filter(lambda x: x["split"] == "dev"),
    })

# ── HF push ───────────────────────────────────────────────────────────────────

def push_to_hub(dataset: DatasetDict, repo_id: str, token: Optional[str], private: bool) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EventStoryLine v0.9 → ECI HF dataset."
    )
    p.add_argument(
        "--root_dir",
        default="data/EventStoryLine/annotated_data/v0.9",
        help="Root folder containing per-topic subfolders of v0.9 XML files.",
    )
    p.add_argument(
        "--repo_id",
        default="Nofing/EventStoryLine-0.9-standard-eci",
        help="HuggingFace dataset repo to push to.",
    )
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (falls back to $HF_TOKEN).")
    p.add_argument(
        "--eval_dir",
        default=None,
        help="Path to evaluation_format/.../event_mentions_extended/ "
             "(auto-derived from root_dir if omitted).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(root_dir=args.root_dir, eval_dir=args.eval_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
