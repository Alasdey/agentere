"""
fix_unannotated.py

For each MECI log file, replace gold="NoRel" with gold="unannotated"
for pairs that do not appear in the HuggingFace dataset's relations field
(i.e., they were never reviewed by annotators).

A pair is considered annotated if it appears in any of:
  relations["CauseEffect"], relations["EffectCause"], relations["NoRel"]
with the same direction as recorded in the log.

Usage:
    python fix_unannotated.py                        # all MECI logs in ../logs/
    python fix_unannotated.py path/to/run.json ...  # specific files
    python fix_unannotated.py --dry-run              # print stats, no writes
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from datasets import load_dataset

REPO_ID = "Nofing/MECI-v0.1-public-span"
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def build_annotated_pairs(ds) -> dict[str, set[tuple[str, str]]]:
    """
    Returns a dict mapping doc_id → set of (src_mention_id, tgt_mention_id)
    for every pair that appears in any relation type (including NoRel).
    """
    annotated: dict[str, set[tuple[str, str]]] = {}
    for row in ds:
        doc_id = row["id"]
        mentions = row["mentions"]
        relations = row.get("relations", {})

        pairs: set[tuple[str, str]] = set()
        for rel_pairs in relations.values():
            for i, j in rel_pairs:
                if i < len(mentions) and j < len(mentions):
                    pairs.add((mentions[i], mentions[j]))

        annotated[doc_id] = pairs
    return annotated


def fix_log(path: str, annotated: dict[str, set[tuple[str, str]]], dry_run: bool) -> tuple[int, int]:
    """
    Processes a single log JSON file in-place.
    Returns (total_norel, changed) counts.
    """
    with open(path) as f:
        data = json.load(f)

    ppp = data.get("results", {}).get("per_pair_predictions")
    if not isinstance(ppp, list):
        return 0, 0

    total_norel = 0
    changed = 0

    for entry in ppp:
        if entry.get("gold") != "NoRel":
            continue
        total_norel += 1

        doc_id = entry.get("id", "")
        pair_str = entry.get("pair", "")
        if not pair_str:
            continue

        parts = pair_str.split(",")
        if len(parts) != 2:
            continue
        src, tgt = parts[0].strip(), parts[1].strip()

        doc_pairs = annotated.get(doc_id)
        if doc_pairs is None:
            # doc not found in HF dataset — leave unchanged
            continue

        if (src, tgt) not in doc_pairs:
            entry["gold"] = "unannotated"
            changed += 1

    if not dry_run and changed > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return total_norel, changed


def find_meci_logs(logs_dir: str) -> list[str]:
    paths = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            if cfg.get("active_dataset") == "meci":
                ppp = data.get("results", {}).get("per_pair_predictions")
                if isinstance(ppp, list):
                    paths.append(path)
        except Exception:
            continue
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Log JSON files to process (default: all MECI logs)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without modifying files")
    args = parser.parse_args()

    print(f"Loading dataset {REPO_ID} (test split)...")
    ds = load_dataset(REPO_ID, split="test")
    annotated = build_annotated_pairs(ds)
    print(f"  Loaded {len(annotated)} documents.\n")

    files = args.files if args.files else find_meci_logs(LOGS_DIR)
    if not files:
        print("No MECI log files found.", file=sys.stderr)
        sys.exit(1)

    total_changed = 0
    for path in files:
        norel, changed = fix_log(path, annotated, dry_run=args.dry_run)
        status = "DRY RUN" if args.dry_run else "updated"
        print(f"{'[dry]' if args.dry_run else '[' + status + ']':10s}  {os.path.basename(path)}")
        print(f"             NoRel entries: {norel}  |  changed to unannotated: {changed}")
        total_changed += changed

    print(f"\nTotal changed: {total_changed}" + (" (dry run — no files written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
