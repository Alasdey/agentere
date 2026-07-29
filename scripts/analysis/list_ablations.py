#!/usr/bin/env python3
"""
List ablation studies from MLflow: every pair of runs whose configs are
identical except for exactly ONE parameter.

For each such pair it reports the changed parameter (both values) and each run's
binary intra-sentence metrics: binary_intra_f1 / binary_intra_precision /
binary_intra_recall.

Params come from mlflow.log_params(_flatten_config(config)) in
utils/mlflow_tracker.py, so the config is fully declarative — no per-run noise
params (paths/timestamps/git-sha live in tags & the run name), which makes
"differ by exactly one key" a clean definition of an ablation pair.

Usage (from the agentere env; sqlite path resolves relative to the project root):
    uv run python scripts/analysis/list_ablations.py
    uv run python scripts/analysis/list_ablations.py --all          # keep pairs even if a run lacks binary_intra
    uv run python scripts/analysis/list_ablations.py --param model.default_model_id
    uv run python scripts/analysis/list_ablations.py --csv scripts/analysis/out/ablations.csv --json scripts/analysis/out/ablations.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "agentere"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # mirrors config.yaml's mlflow.tracking_uri

# The three binary intra-sentence metrics logged per run.
INTRA_KEYS = ("binary_intra_f1", "binary_intra_precision", "binary_intra_recall")

# Params that vary per run by design and are not part of the config identity.
IGNORE_PARAMS = {"experiment.tracing_name"}

# Params that must be HELD EQUAL for two runs to be comparable, i.e. they can
# never be the single ablated variable. Unlike IGNORE_PARAMS these stay in the
# diff, so a pair differing in one of them is simply not an ablation — this keeps
# runs with a different sample size (dataset.max_examples) from being paired.
HOLD_EQUAL_PARAMS = {"dataset.max_examples"}

# Sentinel for "this param key is absent from this run". A key set in one run but
# not the other is itself a difference (e.g. a feature whose sub-params are only
# logged when it is enabled).
ABSENT = "\x00absent"


def resolve_tracking_uri(uri: str) -> str:
    """Make a relative sqlite path absolute against the project root so the
    script works regardless of the current directory (matches param_ablation.py's
    project-root resolution idiom)."""
    project_root = Path(__file__).resolve().parents[2]
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        db = uri[len(prefix):]
        if not db.startswith("/"):
            db = str(project_root / db)
        return f"sqlite:///{db}"
    if not uri.startswith(("http", "/")):
        return str(project_root / uri)
    return uri


def short(v: str, n: int = 70) -> str:
    """One-line, length-capped rendering for display (prompt params are blobs)."""
    if v == ABSENT:
        return "<absent>"
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


def run_date(run) -> str:
    """UTC start datetime of the run (matches the Z-suffixed run name)."""
    return datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def intra(run) -> dict[str, float | None]:
    """binary_intra metrics for a run; None where a metric was not logged."""
    m = run.data.metrics
    return {k: (m[k] if k in m else None) for k in INTRA_KEYS}


def has_all_intra(run) -> bool:
    return all(k in run.data.metrics for k in INTRA_KEYS)


def is_zero_intra_f1(run) -> bool:
    """True if the run logged binary_intra_f1 and it is exactly 0 (degenerate:
    the model predicted no correct intra-sentence causal pair)."""
    m = run.data.metrics
    return "binary_intra_f1" in m and m["binary_intra_f1"] == 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--experiment", default=EXPERIMENT_NAME)
    ap.add_argument("--vs", default=None, metavar="RUN",
                    help="baseline run (name or id): compare ONLY this run against every other "
                         "run and report those differing by exactly one param (baseline = A)")
    ap.add_argument("--max-diff", type=int, default=1, metavar="N",
                    help="max number of parameters allowed to differ (default 1; use 2 to also "
                         "include two-parameter neighbors). Held-equal params never count.")
    ap.add_argument("--param", default=None,
                    help="restrict to pairs whose changed set includes this param (e.g. few_shot.enabled)")
    ap.add_argument("--all", action="store_true",
                    help="keep pairs even when one/both runs lack binary_intra metrics")
    ap.add_argument("--ignore", nargs="*", default=[],
                    help="extra param keys to ignore when diffing")
    ap.add_argument("--hold-equal", nargs="*", default=[],
                    help="extra param keys that must match (never the ablated variable)")
    ap.add_argument("--csv", default=None, help="write the full table to this CSV path")
    ap.add_argument("--json", default=None, help="write the full table (full param values) to this JSON path")
    args = ap.parse_args()

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"No experiment named '{args.experiment}' found at {tracking_uri}")
        sys.exit(1)

    raw_runs = [r for r in client.search_runs([exp.experiment_id], max_results=50_000) if r.data.params]

    baseline = None
    if args.vs:
        baseline = next((r for r in raw_runs if args.vs in (r.info.run_id, r.info.run_name)), None)
        if baseline is None:
            print(f"Baseline run '{args.vs}' not found in experiment '{args.experiment}'.")
            sys.exit(1)

    # Drop degenerate runs (binary_intra_f1 == 0) before pairing — but never the
    # explicitly chosen baseline.
    runs = [r for r in raw_runs
            if not is_zero_intra_f1(r) or (baseline and r.info.run_id == baseline.info.run_id)]
    dropped_zero = len(raw_runs) - len(runs)

    ignore = IGNORE_PARAMS | set(args.ignore)
    hold_equal = HOLD_EQUAL_PARAMS | set(args.hold_equal)
    all_keys = set().union(*(set(r.data.params) for r in runs)) - ignore

    # Per-run param vector over the union of keys (ABSENT where a key is unset).
    vec = {r.info.run_id: {k: r.data.params.get(k, ABSENT) for k in all_keys} for r in runs}

    # --vs: compare only baseline against every other run (baseline is always A).
    # Otherwise: compare every unordered pair (oldest run is A).
    if baseline:
        candidates = ((baseline, b) for b in runs if b.info.run_id != baseline.info.run_id)
    else:
        candidates = itertools.combinations(runs, 2)

    pairs = []
    skipped_no_intra = 0
    for a, b in candidates:
        va, vb = vec[a.info.run_id], vec[b.info.run_id]
        diffs = [k for k in all_keys if va[k] != vb[k]]
        if not 1 <= len(diffs) <= args.max_diff:
            continue
        if any(k in hold_equal for k in diffs):  # a held-equal knob differs → not comparable
            continue
        if args.param and args.param not in diffs:
            continue
        if not args.all and not (has_all_intra(a) and has_all_intra(b)):
            skipped_no_intra += 1
            continue
        run_a, run_b = (a, b) if baseline else sorted((a, b), key=lambda r: r.info.start_time)
        ra, rb = vec[run_a.info.run_id], vec[run_b.info.run_id]
        changes = [{"param": k, "value_a": ra[k], "value_b": rb[k]} for k in sorted(diffs)]
        pairs.append({"changes": changes, "n": len(diffs), "run_a": run_a, "run_b": run_b})

    def df1(p):
        fa, fb = intra(p["run_a"])["binary_intra_f1"], intra(p["run_b"])["binary_intra_f1"]
        return None if fa is None or fb is None else fb - fa

    def changed_names(p):
        return tuple(c["param"] for c in p["changes"])

    def group_label(p):
        # 1 changed param → group by that param's name (keeps single-param sections);
        # >1 → one combined section per count.
        return changed_names(p)[0] if p["n"] == 1 else f"{p['n']} changed parameters"

    # Fewest changes first; within that, single-param grouped by name, multi by |Δf1|.
    pairs.sort(key=lambda p: (p["n"], group_label(p) if p["n"] == 1 else "",
                              -(abs(df1(p)) if df1(p) is not None else -1)))

    # ── Console report ──────────────────────────────────────────────────────
    print(f"Experiment '{args.experiment}' @ {tracking_uri}")
    if baseline:
        bi = intra(baseline)["binary_intra_f1"]
        print(f"Baseline A: {baseline.info.run_name}  [{baseline.info.run_id[:8]}]  "
              f"({run_date(baseline)})  binary_intra_f1={bi:.4f}" if bi is not None
              else f"Baseline A: {baseline.info.run_name}  [{baseline.info.run_id[:8]}]  ({run_date(baseline)})")
    print(f"{len(runs)} usable runs ({dropped_zero} dropped for binary_intra_f1==0) | "
          f"{len(all_keys)} param keys | ignoring {sorted(ignore)} | holding equal {sorted(hold_equal)}")
    print(f"{len(pairs)} ablation pairs (≤{args.max_diff} changed param{'s' if args.max_diff > 1 else ''})"
          + (" vs baseline" if baseline else "")
          + ("" if args.all else f" (+{skipped_no_intra} skipped: a run lacked binary_intra; pass --all to keep)"))
    print()

    counts = Counter(p["n"] for p in pairs)
    for n in sorted(counts):
        print(f"  {counts[n]:>3}  pairs with {n} changed param{'s' if n > 1 else ''}")

    cur = None
    for p in pairs:
        gl = group_label(p)
        if gl != cur:
            cur = gl
            print(f"\n═══ {gl} ═══")
        ma, mb, d = intra(p["run_a"]), intra(p["run_b"]), df1(p)
        dstr = f"  Δf1={d:+.4f}" if d is not None else ""
        if p["n"] == 1:
            c = p["changes"][0]
            line = f"{short(c['value_a'])}   →   {short(c['value_b'])}"
        else:
            line = "   ;   ".join(
                f"{c['param']}: {short(c['value_a'], 34)} → {short(c['value_b'], 34)}" for c in p["changes"])
        print(f"\n  {line}{dstr}")
        for tag, r, m in (("A", p["run_a"], ma), ("B", p["run_b"], mb)):
            fmt = lambda x: "  n/a " if x is None else f"{x:.4f}"
            print(f"      {tag} {run_date(r)}  "
                  f"intra_f1={fmt(m['binary_intra_f1'])}  "
                  f"p={fmt(m['binary_intra_precision'])}  "
                  f"r={fmt(m['binary_intra_recall'])}  [{r.info.run_id[:8]}]")

    # ── Optional file exports ───────────────────────────────────────────────
    def disp(v):
        return None if v == ABSENT else v

    rows = []
    for p in pairs:
        ma, mb = intra(p["run_a"]), intra(p["run_b"])
        rows.append({
            "n_changed": p["n"],
            "changed_params": " ; ".join(
                f"{c['param']}={disp(c['value_a'])}→{disp(c['value_b'])}" for c in p["changes"]),
            "changes": [{"param": c["param"], "value_a": disp(c["value_a"]), "value_b": disp(c["value_b"])}
                        for c in p["changes"]],
            "run_a_date": run_date(p["run_a"]),
            "run_b_date": run_date(p["run_b"]),
            "run_a_name": p["run_a"].info.run_name,
            "run_b_name": p["run_b"].info.run_name,
            "run_a_id": p["run_a"].info.run_id,
            "run_b_id": p["run_b"].info.run_id,
            "a_binary_intra_f1": ma["binary_intra_f1"],
            "a_binary_intra_precision": ma["binary_intra_precision"],
            "a_binary_intra_recall": ma["binary_intra_recall"],
            "b_binary_intra_f1": mb["binary_intra_f1"],
            "b_binary_intra_precision": mb["binary_intra_precision"],
            "b_binary_intra_recall": mb["binary_intra_recall"],
            "delta_binary_intra_f1": df1(p),
        })

    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in rows[0] if k != "changes"]  # nested list: JSON only
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows → {args.csv}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"wrote {len(rows)} rows → {args.json}")


if __name__ == "__main__":
    main()
