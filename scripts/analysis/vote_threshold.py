#!/usr/bin/env python3
"""
Vote-threshold sweep (union / majority / unanimous / any t <= N) for an MLflow run.

For a run with N resampling passes, re-scores the binary causal decision under every
vote threshold t in 1..N ("predict causal iff >= t passes said causal") and reports
P/R/F1 on the all / intra / inter pair subsets, next to the run's logged metrics.

Two sources:
  traces (default): re-pools per-pass predictions from the run's <run>.traces.jsonl
      artifact, matching documents against the HF dataset named in the run's params.
      Works for ALL runs. Validated: reproduces the logged majority metrics within
      ~1 F1 on the 2026-07 final runs.
  log: uses per_pair_predictions vote_counts from the <run>.json artifact.
      ONLY valid for runs logged after the 2026-07-19 utils/reporting.py fix —
      earlier logs are missing voted-but-rejected pairs (the union FPs), which
      silently inflates union precision. The script refuses pre-fix logs.

Usage:
  python scripts/analysis/vote_threshold.py <run>            # uuid prefix or name substring
  python scripts/analysis/vote_threshold.py 204db64f
  python scripts/analysis/vote_threshold.py run_2026-07-19T00-14 --source log
  python scripts/analysis/vote_threshold.py 204db64f --json out.json

Run from the repo root with the project venv (needs datasets + the repo's dataprep).
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAUSAL = {"causes", "causedby"}

# Distinctive prefixes of the CoT-generation hint prompts (tools/few_shot.py).
# Any trace line whose human turns contain one of these is a demo-synthesis call,
# not an inference pass. Imported lazily with a hardcoded fallback so the script
# keeps working if the constants move.
def _cot_markers():
    try:
        from tools import few_shot as fs
        return tuple(
            s.split("{")[0]
            for s in (fs._COT_GOLD_HINT, fs._COT_GOLD_HINT_CONTINUE,
                      fs._COT_CORRECT_HINT, fs._COT_DECONTAM_PROMPT)
        )
    except Exception:
        return (
            "The correct label assignments for this document are:",
            "Continue. The correct labels are still:",
            "Rewrite your response from scratch to remove all privileged knowledge.",
        )


# ---------------------------------------------------------------------------
# MLflow run resolution
# ---------------------------------------------------------------------------

def resolve_run(db_path: Path, selector: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT run_uuid, experiment_id, name FROM runs "
        "WHERE lifecycle_stage='active' AND (run_uuid LIKE ? OR name LIKE ?)",
        (selector + "%", "%" + selector + "%"),
    ).fetchall()
    if not rows:
        raise SystemExit(f"No active run matches '{selector}'")
    if len(rows) > 1:
        listing = "\n".join(f"  {u[:8]}  {n}" for u, _, n in rows[:10])
        raise SystemExit(f"Selector '{selector}' is ambiguous ({len(rows)} runs):\n{listing}")
    run_uuid, exp_id, name = rows[0]
    params = dict(cur.execute(
        "SELECT key, value FROM params WHERE run_uuid=?", (run_uuid,)).fetchall())
    metrics = dict(cur.execute(
        "SELECT key, value FROM metrics WHERE run_uuid=?", (run_uuid,)).fetchall())
    con.close()
    art = db_path.parent / "mlruns" / str(exp_id) / run_uuid / "artifacts"
    if not art.is_dir():
        raise SystemExit(f"Artifacts dir not found: {art}")
    return run_uuid, name, params, metrics, art


def load_docs(params):
    from dataprep.dataprep import load_hf_dataset_parsed
    return list(load_hf_dataset_parsed(
        repo_id=params["dataset.repo_id"],
        split=params["dataset.split"],
        text_field=params["dataset.text_field"],
        ann_field=params["dataset.ann_field"],
        valid_labels=set(CAUSAL) | {"norel"},
        max_examples=int(params["dataset.max_examples"]),
    ))


# ---------------------------------------------------------------------------
# Per-pass vote extraction
# ---------------------------------------------------------------------------

def parse_ai_json(text):
    """Ordered causal pairs from the last JSON array in a response; None if unparseable."""
    blocks = (re.findall(r"```json\s*(\[.*?\])\s*```", text, re.S)
              or re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.S))
    if not blocks:
        return None
    try:
        arr = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None
    out = set()
    for e in arr:
        try:
            a, b = (x.strip() for x in e["pair"].split(","))
            lbl = e["label"].strip().lower()
        except (KeyError, AttributeError, ValueError, TypeError):
            continue
        if lbl in CAUSAL:
            out.add((a, b))
    return out


def votes_from_traces(art_dir: Path, params, docs, n_runs):
    trace_files = sorted(art_dir.glob("*.traces.jsonl"))
    if not trace_files:
        raise SystemExit(f"No .traces.jsonl artifact in {art_dir}")
    markers = _cot_markers()
    few_shot_on = params.get("few_shot.enabled") == "True"
    prefix = {d["doc_text"][:120]: d for d in docs}

    passes = defaultdict(list)
    n_lines = n_gen = n_unmatched = n_unparsed = 0
    with open(trace_files[0], encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)
            n_lines += 1
            humans = [m["kwargs"]["content"] for m in msgs
                      if m["id"][-1] == "HumanMessage"]
            if any(mk in h for mk in markers for h in humans):
                n_gen += 1          # gold-guided / correction / decontamination call
                continue
            # 3-step CoT synthesis starts with a blind call indistinguishable from a
            # no-few-shot inference call; with few-shot on, real inference lines carry
            # the injected demo turns, so 3-message lines can only be blind calls.
            if few_shot_on and len(msgs) <= 3:
                n_gen += 1
                continue
            # Reverse order: few-shot demo turns contain full documents from the
            # same corpus and precede the eval-doc turn — the last matching human
            # is the document actually being classified.
            doc = next((d for h in reversed(humans)
                        for p, d in prefix.items() if p in h), None)
            if doc is None:
                n_unmatched += 1
                continue
            if msgs[-1]["id"][-1] != "AIMessage":
                n_unparsed += 1
                continue
            preds = parse_ai_json(msgs[-1]["kwargs"]["content"])
            if preds is None:
                n_unparsed += 1
                continue
            passes[doc["id"]].append(preds)

    print(f"traces: {n_lines} lines -> {n_gen} CoT-generation, "
          f"{n_unmatched} unmatched, {n_unparsed} unparseable, "
          f"{sum(len(v) for v in passes.values())} inference passes over {len(passes)} docs")
    hist = Counter(min(len(v), n_runs) for v in passes.values())
    if hist and set(hist) != {n_runs}:
        print(f"  note: passes/doc kept (capped at n_runs={n_runs}): {dict(hist)}")

    votes = {}
    for doc_id, runs in passes.items():
        c = Counter()
        for pred in runs[-n_runs:]:     # retries precede the accepted pass
            for pr in pred:
                c[pr] += 1
        votes[doc_id] = c
    return votes


def votes_from_log(art_dir: Path):
    reports = sorted(p for p in art_dir.glob("*.json") if "traces" not in p.name)
    if not reports:
        raise SystemExit(f"No report .json artifact in {art_dir}")
    rows = json.load(open(reports[0]))["results"]["per_pair_predictions"]

    def causal_votes(r):
        return sum(v for k, v in (r.get("vote_counts") or {}).items() if k in CAUSAL)

    if not any(r["gold"] == "norel" and r["pred"] == "norel" and causal_votes(r) > 0
               for r in rows):
        raise SystemExit(
            "This report has no voted-but-rejected rows: it predates the 2026-07-19 "
            "utils/reporting.py fix, so union false positives are missing and any "
            "threshold sweep on it would be silently wrong. Use --source traces.")
    votes = defaultdict(Counter)
    for r in rows:
        a, b = (x.strip() for x in r["pair"].split(","))
        votes[r["id"]][(a, b)] = causal_votes(r)
    return votes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sweep(docs, votes, n_runs):
    """{threshold: {subset: (P, R, F1, tp, fp, fn)}} on ordered pairs (harness convention)."""
    def rel(ms, a, b):
        sa, sb = ms.get(a), ms.get(b)
        if sa is None or sb is None:
            return "unknown"
        return "intra" if sa == sb else "inter"

    out = {}
    for t in range(1, n_runs + 1):
        agg = {s: [0, 0, 0] for s in ("all", "intra", "inter")}
        for d in docs:
            ms = d.get("mention_sentence") or {}
            gold = {(s, tg): rel(ms, s, tg)
                    for s, lbl, tg in d["gold_triples"] if lbl in CAUSAL}
            pred = {pr: rel(ms, *pr)
                    for pr, v in votes.get(d["id"], {}).items() if v >= t}
            for pair in set(gold) | set(pred):
                subset = gold.get(pair, pred.get(pair))
                g, p = pair in gold, pair in pred
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
    ap.add_argument("--source", choices=["traces", "log"], default="traces")
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
          f"n_runs={n_runs} source={args.source}")

    docs = load_docs(params)
    votes = (votes_from_traces(art, params, docs, n_runs) if args.source == "traces"
             else votes_from_log(art))
    result = sweep(docs, votes, n_runs)

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
        if abs(F - 100 * logged["binary_f1"]) > 2.5:
            print("WARNING: majority reconstruction deviates >2.5 F1 from the logged "
                  "metric — doc matching or pass selection may be off for this run; "
                  "treat the sweep with caution.")

    if args.json_out:
        payload = {
            "run_uuid": run_uuid, "run_name": name, "n_runs": n_runs,
            "source": args.source, "shipped_threshold": t_shipped,
            "thresholds": {t: {s: dict(zip(("P", "R", "F1", "tp", "fp", "fn"), v))
                               for s, v in subs.items()}
                           for t, subs in result.items()},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
