"""
run_queue.py — run queued experiments sequentially.

Usage:
  python run_queue.py                       # run all experiments in queue.yaml
  python run_queue.py --only exp1 exp2      # run only named experiments
  python run_queue.py --queue my_queue.yaml # use a different queue file
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import yaml

from main import main
from utils.config import load_config_with_overrides


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="queue.yaml", help="Queue file (default: queue.yaml)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Run only these named experiments")
    return parser.parse_args()


def load_queue(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["experiments"]


def main_sync():
    args = parse_args()
    experiments = load_queue(args.queue)

    if args.only:
        names = set(args.only)
        experiments = [e for e in experiments if e["name"] in names]
        missing = names - {e["name"] for e in experiments}
        if missing:
            print(f"ERROR: experiments not found in queue: {missing}", file=sys.stderr)
            sys.exit(1)

    print(f"Queue: {len(experiments)} experiment(s) to run")
    print()

    total_start = time.time()
    failed = []

    for i, entry in enumerate(experiments, 1):
        name = entry["name"]
        overrides = entry.get("overrides", {})
        print(f"{'='*60}")
        print(f"[{i}/{len(experiments)}] Starting: {name}")
        print(yaml.dump(overrides, default_flow_style=False).rstrip())
        print(f"{'='*60}")

        try:
            config = load_config_with_overrides(overrides)
            t0 = time.time()
            asyncio.run(main(config=config))
            elapsed = time.time() - t0
            print(f"\n[{name}] Done in {elapsed:.0f}s\n")
        except Exception as e:
            elapsed = time.time() - t0 if 't0' in dir() else 0
            print(f"\n[{name}] FAILED after {elapsed:.0f}s: {e}\n", file=sys.stderr)
            failed.append(name)

    total_elapsed = time.time() - total_start
    print(f"{'='*60}")
    print(f"Queue finished in {total_elapsed:.0f}s — {len(experiments) - len(failed)}/{len(experiments)} succeeded")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
