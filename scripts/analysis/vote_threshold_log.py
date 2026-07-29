#!/usr/bin/env python3
"""
Vote-threshold sweep from the run's logged per-pair predictions ONLY (no traces, no HF).

Log-native counterpart to scripts/analysis/vote_threshold.py. For a run with N
resampling passes, re-scores the binary causal decision under every vote threshold
t in 1..N ("predict causal iff >= t passes said causal") and reports P/R/F1 on the
all / intra / inter subsets, next to the run's logged metrics.

Why a separate script (and when to use which):
  This reads everything from the <run>.json artifact's per_pair_predictions rows —
  each row carries vote_counts, gold, and sentence_relation. So it needs no HF
  download, no dataprep, no datasets, and no trace re-parsing. It is exact: the
  vote tallies are the pipeline's own (aggregate_run_triples), not a best-effort
  re-parse of raw traces, so there are no unparseable-line losses and no
  document-matching heuristics.

  It ONLY works for runs logged after the 2026-07-19 utils/reporting.py fix, which
  started logging every voted pair (not just the finally-voted-causal ones) with
  full vote_counts. Earlier logs are missing the voted-but-rejected pairs (union
  FPs) and their non-causal vote counts, so the sweep would be silently wrong.
  The script refuses such logs — use vote_threshold.py --source traces for them.

  Empirically (run c5490b4e, n_runs=3): the majority reconstruction lands within
  ~0.4 F1 of the logged metric on both all and intra, versus ~1.4 / ~3.1 F1 for the
  trace-reconstruction path — the log source removes the data-quality noise, leaving
  only the small definitional gap between this directed causal-vote sweep and the
  shipped plurality/undirected-merge aggregation.

Usage:
  python scripts/analysis/vote_threshold_log.py <run>        # uuid prefix or name substring
  python scripts/analysis/vote_threshold_log.py 883c3fd0
  python scripts/analysis/vote_threshold_log.py 883c3fd0 --json out.json

Run from the repo root; no venv extras needed (stdlib + sqlite3 only).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Run resolution (mlflow.db lookup + artifacts dir) is shared with the trace-based
# script so the two stay consistent; importing does not pull in HF/dataprep.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote_threshold import resolve_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CAUSAL = {"causes", "causedby"}


def causal_votes(vote_counts):
    return sum(v for k, v in (vote_counts or {}).items() if k in CAUSAL)


def load_rows(art_dir: Path):
    reports = sorted(p for p in art_dir.glob("*.json") if "traces" not in p.name)
    if not reports:
        raise SystemExit(f"No report .json artifact in {art_dir}")
    return json.load(open(reports[0], encoding="utf-8"))["results"]["per_pair_predictions"]


def require_postfix(rows):
    """Refuse pre-2026-07-19 logs, which lack the voted-but-rejected (union FP) rows."""
    if not any(r["gold"] == "norel" and r["pred"] == "norel"
               and causal_votes(r.get("vote_counts")) > 0 for r in rows):
        raise SystemExit(
            "This report has no voted-but-rejected rows: it predates the 2026-07-19 "
            "utils/reporting.py fix, so union false positives (and non-causal vote "
            "counts) are missing and a threshold sweep on it would be silently wrong. "
            "Use scripts/analysis/vote_threshold.py --source traces for pre-fix runs.")


def sweep(rows, n_runs):
    """{threshold: {subset: (P, R, F1, tp, fp, fn)}} on ordered pairs (harness convention).

    Each row is one ordered pair from the log's gold ∪ pred ∪ any-voted universe, so
    iterating rows is exactly the trace script's `set(gold) | set(pred_at_t)` universe:
    rows that are neither gold-causal nor predicted-causal at t are true negatives and
    contribute nothing. gold and intra/inter come straight off the row.
    """
    out = {}
    for t in range(1, n_runs + 1):
        agg = {s: [0, 0, 0] for s in ("all", "intra", "inter")}
        for r in rows:
            g = r["gold"] in CAUSAL
            p = causal_votes(r.get("vote_counts")) >= t
            if not (g or p):
                continue
            subset = r["sentence_relation"]
            for s in ("all", subset):
                if s == "unknown":
                    continue
                if p and g:
                    agg[s][0] += 1
                elif p:
                    agg[s][1] += 1
                else:
                    agg[s][2] += 1
        out[t] = {}
        for s, (tp, fp, fn) in agg.items():
            P = 100 * tp / (tp + fp) if tp + fp else 0.0
            R = 100 * tp / (tp + fn) if tp + fn else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            out[t][s] = (P, R, F, tp, fp, fn)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run_uuid prefix or run-name substring")
    ap.add_argument("--db", default=str(REPO_ROOT / "mlflow.db"))
    ap.add_argument("--n-runs", type=int, default=0,
                    help="override number of resample passes (default: run param)")
    ap.add_argument("--json", dest="json_out", default="",
                    help="also write the sweep to this JSON file")
    args = ap.parse_args()

    run_uuid, name, params, metrics, art = resolve_run(Path(args.db).resolve(), args.run)
    n_runs = args.n_runs or (
        int(params["experiment.resampling.n_runs"])
        if params.get("experiment.resampling.enabled") == "True" else 1)
    print(f"run {run_uuid[:8]} ({name})")
    print(f"dataset={params.get('dataset.name')} split={params.get('dataset.split')} "
          f"n={params.get('dataset.max_examples')} model={params.get('model.default_model_id')} "
          f"n_runs={n_runs} source=log")

    rows = load_rows(art)
    require_postfix(rows)

    docs = {r["id"] for r in rows}
    voted = [r for r in rows if r.get("vote_counts")]
    over = sum(1 for r in voted if sum(r["vote_counts"].values()) > n_runs)
    print(f"log: {len(rows)} pair rows over {len(docs)} docs, "
          f"{len(voted)} with vote_counts"
          + (f"; note: {over} rows have >n_runs total votes "
             f"(within-run duplicate pairs) — harmless for a >=t sweep" if over else ""))

    result = sweep(rows, n_runs)

    t_shipped = n_runs // 2 + 1
    print(f"\n{'t':>3} {'':12} {'ALL  P / R / F1':>22} {'INTRA  P / R / F1':>24} "
          f"{'INTER  P / R / F1':>24}")
    for t, subs in result.items():
        label = {1: "(union)", n_runs: "(unanimous)"}.get(t, "")
        if t == t_shipped:
            label = "(majority)*"
        cells = "".join(
            f"  {subs[s][0]:6.1f}/{subs[s][1]:5.1f}/{subs[s][2]:5.1f}"
            for s in ("all", "intra", "inter"))
        print(f">={t:>2} {label:12}{cells}")
    print("* = the shipped aggregation (plurality vote, ties -> norel) for odd n_runs")

    logged = {k: metrics.get(k) for k in
              ("binary_precision", "binary_recall", "binary_f1",
               "binary_intra_precision", "binary_intra_recall", "binary_intra_f1")}
    if logged.get("binary_f1") is not None:
        P, R, F, *_ = result[t_shipped]["all"]
        print(f"\nsanity vs logged run metrics (majority): "
              f"all {P:.1f}/{R:.1f}/{F:.1f} vs logged "
              f"{100*logged['binary_precision']:.1f}/{100*logged['binary_recall']:.1f}/"
              f"{100*logged['binary_f1']:.1f}", end="")
        if logged.get("binary_intra_f1") is not None:
            Pi, Ri, Fi, *_ = result[t_shipped]["intra"]
            print(f" | intra {Pi:.1f}/{Ri:.1f}/{Fi:.1f} vs logged "
                  f"{100*logged['binary_intra_precision']:.1f}/"
                  f"{100*logged['binary_intra_recall']:.1f}/"
                  f"{100*logged['binary_intra_f1']:.1f}", end="")
        print()

    if args.json_out:
        payload = {
            "run_uuid": run_uuid, "run_name": name, "n_runs": n_runs,
            "source": "log", "shipped_threshold": t_shipped,
            "thresholds": {t: {s: dict(zip(("P", "R", "F1", "tp", "fp", "fn"), v))
                               for s, v in subs.items()}
                           for t, subs in result.items()},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
