#!/usr/bin/env python3
"""
Re-score a finished run as if data.vote_to_binary had been enabled.

data.vote_to_binary (config.yaml) is a post-hoc scoring mode: the model still predicts in
directed causes/causedby/norel space, but before metrics the directed vote tallies for
(a,b) and (b,a) are merged into a single binary causal/norel decision per *unordered* set
{a,b}, and gold is collapsed the same way. Each set is therefore evaluated exactly once,
on both the gold and the prediction side.

This reproduces that scoring offline from a finished run's log — no LLM calls, no HF
download, no writes — and reports the full metric set (per-label, macro/micro, binary
all/intra/inter, per-language, per-document), so a run scored in directed space can be
read as if vote_to_binary had been on.

The merge is NOT an OR over directions: causes+causedby votes summed over both directions
are weighed against norel votes summed over both directions, majority wins, ties go to
binary_default. With novote_norel a direction that was never queried adds one implicit
norel vote, so a lone causal vote is a tie too. Both knobs move the numbers, so all four
(binary_default x novote_norel) combinations are swept, next to the direction-agnostic
OR / AND baselines from binary_eval.py.

The vote merge itself is utils.resample.aggregate_votes_to_binary — the production
function the pipeline runs — so this cannot drift from what a live run would report.

ONLY works on runs logged after the 2026-07-19 utils/reporting.py fix, which started
logging every voted pair with its full vote_counts. Earlier logs are missing the
voted-but-rejected rows, so the merge would be silently wrong; such logs are refused.

Usage:
  python scripts/analysis/vote_to_binary_eval.py <run>        # uuid prefix or name substring
  python scripts/analysis/vote_to_binary_eval.py logs/allatonce/run_2026-07-29T11-57-57.330249Z_4085f6ca.json
  python scripts/analysis/vote_to_binary_eval.py 4085f6ca --all-rules
  python scripts/analysis/vote_to_binary_eval.py 4085f6ca --per-doc

Run from the repo root; stdlib + sqlite3 only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# Run resolution and the OR/AND collapse are shared with the sibling analysis scripts so
# the numbers stay directly comparable; neither import pulls in HF/dataprep.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vote_threshold import resolve_run                       # noqa: E402
from vote_threshold_log import require_postfix               # noqa: E402
from binary_eval import aggregate_pair, binarize             # noqa: E402

from utils.labels import BINARY_LABELS, BINARY_POS, NOREL, NOREL_VARIANTS  # noqa: E402
from utils.metrics import compute_binary_metrics, compute_multiclass_metrics  # noqa: E402
from utils.reporting import _split_intra_inter               # noqa: E402
from utils.resample import aggregate_votes_to_binary         # noqa: E402

# Row fields this script depends on. sentence_relation and vote_counts both arrived with
# the reporting rewrite; a log missing either cannot be re-scored.
REQUIRED_ROW_KEYS = ("id", "doc_idx", "lang", "pair", "gold", "pred",
                     "vote_counts", "sentence_relation")


# =============================================================================
# LOADING
# =============================================================================

def load_report(selector: str, db_path: Path):
    """Returns (label, report_dict) for a run selector or a direct path to a run JSON."""
    as_path = Path(selector)
    if as_path.is_file():
        report = json.loads(as_path.read_text(encoding="utf-8"))
        return as_path.name, report

    run_uuid, name, _params, _metrics, art = resolve_run(db_path, selector)
    reports = sorted(p for p in art.glob("*.json") if "traces" not in p.name)
    if not reports:
        raise SystemExit(f"No report .json artifact in {art}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    return f"{run_uuid[:8]} ({name})", report


def validate(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Refuses logs this script cannot re-score, and returns the pair rows."""
    rows = report["results"]["per_pair_predictions"]
    if not rows:
        raise SystemExit("Report has an empty per_pair_predictions list.")

    missing = [k for k in REQUIRED_ROW_KEYS if k not in rows[0]]
    if missing:
        raise SystemExit(
            f"per_pair_predictions rows are missing {missing}: this log predates the "
            "2026-07-19 utils/reporting.py rewrite and cannot be re-scored.")

    # Every setting read below must be present; a config too old to carry them is a
    # config too old to re-score against.
    missing_cfg = [k for k in ("vote_to_binary", "binary_default", "novote_norel")
                   if k not in report["config"]["data"]]
    if missing_cfg:
        raise SystemExit(
            f"config.data is missing {missing_cfg}: this run predates those settings, so "
            "there is no configured vote_to_binary behaviour to reproduce.")

    if report["config"]["data"]["vote_to_binary"]:
        raise SystemExit(
            "This run already ran with data.vote_to_binary=true: its per_pair_predictions "
            "rows are the merged undirected decisions, not the directed ones, so there is "
            "nothing left to merge. Read the run's own metrics instead.")

    # Refuses pre-fix logs, which lack the voted-but-rejected rows (and hence the
    # non-causal vote counts the undirected merge needs).
    require_postfix(rows)
    return rows


# =============================================================================
# RECONSTRUCTION
# =============================================================================

class DocView:
    """The per-document state vote_to_binary scoring needs, rebuilt from log rows.

    pair_stats is a faithful reconstruction of the live pipeline's pair_stats dict
    (main.py) — the exact input aggregate_votes_to_binary consumes.
    """

    __slots__ = ("doc_id", "doc_idx", "lang", "pair_stats", "gold_sets",
                 "directed_pred", "sent_rel")

    def __init__(self, doc_id: str, doc_idx: int, lang: str):
        self.doc_id = doc_id
        self.doc_idx = doc_idx
        self.lang = lang
        self.pair_stats: Dict[str, dict] = {}
        self.gold_sets: Set[FrozenSet[str]] = set()
        self.directed_pred: Dict[FrozenSet[str], List[str]] = defaultdict(list)
        self.sent_rel: Dict[FrozenSet[str], str] = {}

    @property
    def all_sets(self) -> Set[FrozenSet[str]]:
        """Every unordered pair the log knows about — the evaluation universe.

        Equals gold_sets | voted_sets: predictions can only come from voted pairs, so
        this is the undirected image of the live gold | pred | voted universe.
        """
        return set(self.sent_rel)


def build_docs(rows: List[Dict[str, Any]]) -> Dict[str, DocView]:
    docs: Dict[str, DocView] = {}
    sent_conflicts = 0

    for row in rows:
        doc_id = row["id"]
        if doc_id not in docs:
            docs[doc_id] = DocView(doc_id, row["doc_idx"], row["lang"])
        doc = docs[doc_id]

        src, tgt = (p.strip() for p in row["pair"].split(",", 1))
        key = frozenset((src, tgt))

        # vote_counts is None for a pair that was never voted on (a gold pair the model
        # never emitted). Those have no pair_stats entry in a live run either.
        if row["vote_counts"]:
            doc.pair_stats[row["pair"]] = {"vote_counts": row["vote_counts"]}

        if row["gold"].lower() not in NOREL_VARIANTS:
            doc.gold_sets.add(key)

        doc.directed_pred[key].append(binarize(row["pred"]))

        # Symmetric by construction (mention_sentence lookup ignores direction), so the
        # two directions must agree; count any disagreement instead of silently picking.
        rel = row["sentence_relation"]
        if key in doc.sent_rel and doc.sent_rel[key] != rel:
            sent_conflicts += 1
        doc.sent_rel[key] = rel

    if sent_conflicts:
        print(f"WARNING: {sent_conflicts} undirected pairs had disagreeing "
              f"sentence_relation across their two directions", file=sys.stderr)
    return docs


# =============================================================================
# COLLAPSE RULES
# =============================================================================

def vote_rule(binary_default: str, novote_norel: bool) -> Callable[[DocView], Set[FrozenSet[str]]]:
    """The shipped vote_to_binary merge, via the production aggregator."""
    def positives(doc: DocView) -> Set[FrozenSet[str]]:
        return {
            frozenset((src, tgt))
            for src, _lbl, tgt in aggregate_votes_to_binary(
                doc.pair_stats, binary_default=binary_default, novote_norel=novote_norel)
        }
    return positives


def directed_rule(mode: str) -> Callable[[DocView], Set[FrozenSet[str]]]:
    """binary_eval.py's direction-agnostic baselines over the run's final directed labels."""
    def positives(doc: DocView) -> Set[FrozenSet[str]]:
        return {
            key for key, labels in doc.directed_pred.items()
            if aggregate_pair(labels, mode) == "Causal"
        }
    return positives


# =============================================================================
# SCORING — mirrors utils.reporting.generate_run_report at undirected-set granularity
# =============================================================================

def score(docs: Dict[str, DocView],
          positives_fn: Callable[[DocView], Set[FrozenSet[str]]]) -> Dict[str, Any]:
    """Full report for one collapse rule: one evaluated row per unordered pair set."""
    all_true: List[str] = []
    all_pred: List[str] = []
    all_rel: List[str] = []
    per_doc: List[Dict[str, Any]] = []
    per_lang = defaultdict(lambda: {"y_true": [], "y_pred": [], "rel": []})

    for doc in docs.values():
        positives = positives_fn(doc)
        y_true, y_pred, rel = [], [], []
        for key in doc.all_sets:
            y_true.append(BINARY_POS if key in doc.gold_sets else NOREL)
            y_pred.append(BINARY_POS if key in positives else NOREL)
            rel.append(doc.sent_rel[key])

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_rel.extend(rel)
        per_lang[doc.lang]["y_true"].extend(y_true)
        per_lang[doc.lang]["y_pred"].extend(y_pred)
        per_lang[doc.lang]["rel"].extend(rel)

        mc = compute_multiclass_metrics(y_true, y_pred, BINARY_LABELS)
        it, ip, et, ep, unk = _split_intra_inter(y_true, y_pred, rel)
        per_doc.append({
            "doc_idx": doc.doc_idx,
            "id": doc.doc_id,
            "pairs": len(y_true),
            "macro_f1": mc["macro_f1"],
            "micro_f1": mc["micro_f1"],
            "micro_precision": mc["micro_precision"],
            "micro_recall": mc["micro_recall"],
            "per_label": mc["per_label"],
            "binary": compute_binary_metrics(y_true, y_pred),
            "binary_intra": compute_binary_metrics(it, ip),
            "binary_inter": compute_binary_metrics(et, ep),
            "sentence_unknown_pairs": unk,
        })

    global_mc = compute_multiclass_metrics(all_true, all_pred, BINARY_LABELS)
    it, ip, et, ep, unk = _split_intra_inter(all_true, all_pred, all_rel)
    total = len(all_true)

    per_lang_metrics = {}
    for lang, data in per_lang.items():
        if not data["y_true"]:
            continue
        lit, lip, let, lep, lunk = _split_intra_inter(data["y_true"], data["y_pred"], data["rel"])
        per_lang_metrics[lang] = {
            "multiclass": compute_multiclass_metrics(data["y_true"], data["y_pred"], BINARY_LABELS),
            "binary": compute_binary_metrics(data["y_true"], data["y_pred"]),
            "binary_intra": compute_binary_metrics(lit, lip),
            "binary_inter": compute_binary_metrics(let, lep),
            "sentence_unknown_pairs": lunk,
            "total_pairs": len(data["y_true"]),
        }

    return {
        "per_label": global_mc["per_label"],
        "macro_f1": global_mc["macro_f1"],
        "micro_precision": global_mc["micro_precision"],
        "micro_recall": global_mc["micro_recall"],
        "micro_f1": global_mc["micro_f1"],
        "total_pairs": total,
        "binary": compute_binary_metrics(all_true, all_pred),
        "binary_intra": compute_binary_metrics(it, ip),
        "binary_inter": compute_binary_metrics(et, ep),
        "sentence_unknown_pairs": unk,
        "sentence_unknown_pct": (unk / total) if total else 0.0,
        "per_doc_metrics": sorted(per_doc, key=lambda d: d["doc_idx"]),
        "per_lang_metrics": per_lang_metrics,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def pct(v: float) -> str:
    return f"{100 * v:6.2f}"


def prf(m: Dict[str, float]) -> str:
    return f"{pct(m['precision'])}/{pct(m['recall'])}/{pct(m['f1'])}"


RULE_W = 28


def print_sweep(rules: List[tuple], reports: Dict[str, Dict[str, Any]], logged: Dict[str, Any]):
    head = (f"{'rule':<{RULE_W}} {'pairs':>7} {'micro_f1':>9} {'macro_f1':>9} "
            f"{'ALL  P / R / F1':>24} {'INTRA  P / R / F1':>24} {'INTER  P / R / F1':>24} {'gold+':>7}")
    print(head)
    print("-" * len(head))

    def line(name: str, r: Dict[str, Any]):
        print(f"{name:<{RULE_W}} {r['total_pairs']:>7} "
              f"{pct(r['micro_f1']):>9} {pct(r['macro_f1']):>9} "
              f"{prf(r['binary']):>24} {prf(r['binary_intra']):>24} "
              f"{prf(r['binary_inter']):>24} {r['binary']['support_pos']:>7}")

    line("directed (as logged)", logged)
    print("-" * len(head))
    for name, _fn in rules:
        line(name, reports[name])


def print_full(name: str, r: Dict[str, Any], per_doc: bool):
    print(f"\n{'=' * 100}\n{name} — full metrics\n{'=' * 100}")
    print(f"total_pairs (undirected sets) : {r['total_pairs']}")
    print(f"micro_precision / recall / f1 : "
          f"{pct(r['micro_precision'])}/{pct(r['micro_recall'])}/{pct(r['micro_f1'])}")
    print(f"macro_f1                      : {pct(r['macro_f1'])}")
    print(f"sentence_unknown              : {r['sentence_unknown_pairs']} pairs "
          f"({r['sentence_unknown_pct']:.2%}) — excluded from both intra and inter")

    print("\nper_label:")
    print(f"  {'label':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}")
    for label, m in sorted(r["per_label"].items()):
        print(f"  {label:<12} {pct(m['precision']):>10} {pct(m['recall']):>10} "
              f"{pct(m['f1']):>10} {m['support']:>9}")

    print("\nbinary (any relation = positive):")
    print(f"  {'subset':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support_pos':>12}")
    for subset in ("binary", "binary_intra", "binary_inter"):
        m = r[subset]
        print(f"  {subset.replace('binary_', '').replace('binary', 'all'):<12} "
              f"{pct(m['precision']):>10} {pct(m['recall']):>10} {pct(m['f1']):>10} "
              f"{m['support_pos']:>12}")

    print("\nper_lang_metrics:")
    print(f"  {'lang':<14} {'pairs':>7} {'micro_f1':>9} {'macro_f1':>9} "
          f"{'ALL  P / R / F1':>24} {'INTRA  P / R / F1':>24} {'INTER  P / R / F1':>24} {'unk':>5}")
    for lang, lm in sorted(r["per_lang_metrics"].items()):
        mc = lm["multiclass"]
        print(f"  {lang:<14} {lm['total_pairs']:>7} {pct(mc['micro_f1']):>9} "
              f"{pct(mc['macro_f1']):>9} {prf(lm['binary']):>24} "
              f"{prf(lm['binary_intra']):>24} {prf(lm['binary_inter']):>24} "
              f"{lm['sentence_unknown_pairs']:>5}")

    pd = r["per_doc_metrics"]
    micro = [d["micro_f1"] for d in pd]
    print(f"\nper_doc_metrics: {len(pd)} docs | micro_f1 mean {pct(statistics.fmean(micro))} "
          f"median {pct(statistics.median(micro))} "
          f"min {pct(min(micro))} max {pct(max(micro))}")
    if per_doc:
        print(f"  {'doc_idx':>7} {'pairs':>6} {'micro_f1':>9} {'macro_f1':>9} "
              f"{'ALL  P / R / F1':>24} {'id'}")
        for d in pd:
            print(f"  {d['doc_idx']:>7} {d['pairs']:>6} {pct(d['micro_f1']):>9} "
                  f"{pct(d['macro_f1']):>9} {prf(d['binary']):>24} {d['id']}")


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run_uuid prefix, run-name substring, or path to a run JSON")
    ap.add_argument("--db", default=str(REPO_ROOT / "mlflow.db"))
    ap.add_argument("--all-rules", action="store_true",
                    help="print the full metric block for every collapse rule, not just the "
                         "run-configured one")
    ap.add_argument("--per-doc", action="store_true",
                    help="print the full per-document table instead of just its summary")
    args = ap.parse_args()

    label, report = load_report(args.run, Path(args.db).resolve())
    rows = validate(report)

    cfg = report["config"]
    data_cfg = cfg["data"]
    resampling = cfg["experiment"]["resampling"]
    n_runs = resampling["n_runs"] if resampling["enabled"] else 1
    configured = (data_cfg["binary_default"], data_cfg["novote_norel"])

    docs = build_docs(rows)
    n_sets = sum(len(d.all_sets) for d in docs.values())
    n_voted = sum(1 for r in rows if r["vote_counts"])

    print(f"run {label}")
    print(f"dataset={cfg['active_dataset']} model={cfg['model']['default_model_id']} "
          f"n_runs={n_runs} binary_undirected={data_cfg['binary_undirected']}")
    print(f"log: {len(rows)} directed pair rows over {len(docs)} docs "
          f"({n_voted} with vote_counts) -> {n_sets} undirected pair sets")
    print(f"run config: binary_default={data_cfg['binary_default']} "
          f"novote_norel={data_cfg['novote_norel']}  (marked * below)\n")

    rules: List[tuple] = []
    for default in (NOREL, BINARY_POS):
        for novote in (True, False):
            name = f"vote {default:<6} / novote={'on ' if novote else 'off'}"
            if (default, novote) == configured:
                name += "  *"
            rules.append((name, vote_rule(default, novote)))
    rules.append(("OR   (either direction)", directed_rule("or")))
    rules.append(("AND  (both directions)", directed_rule("and")))

    if not any(n.endswith("*") for n, _ in rules):
        raise SystemExit(
            f"config.data.binary_default is {data_cfg['binary_default']!r}, not "
            f"{NOREL!r} or {BINARY_POS!r} — the sweep has no row matching this run's "
            "settings, so there is nothing to mark as the configured result.")

    reports = {name: score(docs, fn) for name, fn in rules}
    print_sweep(rules, reports, report["results"])

    print("\n* = what this run would have reported under its own binary_default / "
          "novote_norel settings.")
    print("OR / AND collapse the run's FINAL directed labels (binary_eval.py semantics); "
          "the vote rules re-merge the raw per-direction vote_counts, which is what "
          "vote_to_binary actually does.")

    to_expand = [n for n, _ in rules] if args.all_rules else \
        [n for n, _ in rules if n.endswith("*")]
    for name in to_expand:
        print_full(name, reports[name], args.per_doc)

    print(f"\n{'-' * 100}")
    print("NOTE  total_pairs above is one row per undirected set. A live vote_to_binary run "
          "logs about twice\n      that, because reconstruct_pairwise_predictions still emits "
          "the reverse-direction row (as a\n      norel/norel true negative). P/R/F1, micro_f1 "
          "and macro_f1 are unaffected by those rows —\n      only total_pairs and "
          "per_label['norel'] differ.")


if __name__ == "__main__":
    main()
