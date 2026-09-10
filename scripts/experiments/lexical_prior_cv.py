"""Trigger-lemma lookup table, evaluated under each dataset's own standard CV protocol.

A lookup table keyed on the pair of trigger lemmas — no document, no context, no model. The
question is not whether it works but what its score says about the benchmark it is scored on.

Protocols, matched to the literature rather than invented here:
  * Causal-TimeBank — 10-fold cross-validation over documents (Liu et al. 2020 lineage;
    see litt/HOTECI/data_modules/preprocessor.py, KFold(n_splits=10) over a document shuffle).
  * EventStoryLine — 5-fold cross-validation over the *documents* of the 20 non-dev topics
    (Gao et al. 2019; ERGO §5 "documents in the remaining 20 topics are employed for a 5-fold
    cross-validation"). Same code, KFold(n_splits=5), document-level.
  * EventStoryLine, topic-grouped — the same 5 folds cut along ECB+ topic lines instead, so no
    topic straddles the split. Not the standard protocol; the control that isolates what the
    standard one leaks.

Scoring is the backoff chain from lexical_prior.py: global -> source/target lemma marginals
-> undirected lemma pair -> directed lemma pair, each level shrunk toward the one below.
Per fold, counts come only from that fold's training documents. Predictions are pooled across
folds into one PR curve, which is then summarised by average precision and by the best F1 on
the pooled curve.

    uv run python scripts/experiments/lexical_prior_cv.py
    uv run python scripts/experiments/lexical_prior_cv.py --repeats 10 --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep.dataprep import load_hf_dataset_parsed  # noqa: E402
from scripts.experiments.lexical_prior import (  # noqa: E402
    DATASETS, LemmaPrior, Pair, best_f1, group_of, lemmatize, pr_curve,
)


def doc_pairs_scoped(doc: dict, scope: str) -> List[Pair]:
    """Candidate ordered pairs of one document, restricted to `scope`.

    `intra` is the scope every ESC/CTB intra-sentence table is reported on (and the one that
    reproduces Dr.ECI's 17,998 / 3,184 evaluation file exactly); `inter` is its complement;
    `all` is the union.
    """
    lemmas, sent = doc["lemmas"], doc["mention_sentence"]
    gold = {(s, t) for s, _lab, t in doc["gold_triples"]}
    out: List[Pair] = []
    for a, b in doc["pair_list_ids"]:
        if a not in lemmas or b not in lemmas:
            continue
        same = sent.get(a, -1) == sent.get(b, -2)
        if (scope == "intra" and not same) or (scope == "inter" and same):
            continue
        out.append((lemmas[a], lemmas[b], (a, b) in gold))
    return out

PROTOCOLS = [
    # (dataset key, label, n_folds, group by topic?)
    ("causal_timebank", "CTB — 10-fold, document", 10, False),
    ("event_story_line", "ESC — 5-fold, document (standard)", 5, False),
    ("event_story_line", "ESC — 5-fold, topic-grouped (control)", 5, True),
]


def folds_of(units: Sequence[str], n_folds: int, seed: int) -> List[set]:
    """`n_folds` disjoint, near-equal held-out sets over the distinct units."""
    keys = sorted(set(units))
    random.Random(seed).shuffle(keys)
    return [set(keys[i::n_folds]) for i in range(n_folds)]


def cross_validate(per_doc: Sequence[List[Pair]], units: Sequence[str],
                   n_folds: int, alpha: float, seed: int) -> dict:
    """One full CV pass: every document is scored exactly once, by a prior that never saw it."""
    pooled: List[Tuple[float, bool]] = []
    tuned: List[Tuple[bool, bool]] = []   # (predicted, gold) at a threshold chosen without test
    seen_pairs = seen_gold = n_pairs = n_gold = 0
    for held in folds_of(units, n_folds, seed):
        tr_units = [u for u in sorted(set(units)) if u not in held]
        train = [p for p, u in zip(per_doc, units) if u not in held]
        prior = LemmaPrior(train, alpha)

        # Nested threshold selection: an inner fifth of the training units is scored by a prior
        # built on the rest of them, and the F1-optimal cut on that inner curve is the operating
        # point applied to the held-out fold. No test label ever informs the threshold.
        inner_held = set(tr_units[::5])
        inner_prior = LemmaPrior([p for p, u in zip(per_doc, units)
                                  if u not in held and u not in inner_held], alpha)
        inner = [(inner_prior.score(a, b), y)
                 for p, u in zip(per_doc, units) if u in inner_held for a, b, y in p]
        thr = 0.0
        if inner:
            _ap, ipts = pr_curve(inner)
            thr = max(ipts, key=lambda x: (2 * x[1] * x[0] / (x[1] + x[0])) if x[0] + x[1] else 0.0)[2]

        for p, u in zip(per_doc, units):
            if u not in held:
                continue
            for a, b, y in p:
                sc = prior.score(a, b)
                pooled.append((sc, y))
                tuned.append((sc >= thr, y))
                n_pairs += 1
                n_gold += y
                if prior.seen(a, b)[0] >= 1:
                    seen_pairs += 1
                    seen_gold += y

    ap, pts = pr_curve(pooled)
    f1, p, r = best_f1(pts)
    tp = sum(1 for pr_, y in tuned if pr_ and y)
    fp = sum(1 for pr_, y in tuned if pr_ and not y)
    tP = tp / (tp + fp) if tp + fp else 0.0
    tR = tp / max(n_gold, 1)
    tF = 2 * tP * tR / (tP + tR) if tP + tR else 0.0
    return {"ap": ap, "f1": f1, "precision": p, "recall": r,
            "tuned_f1": tF, "tuned_precision": tP, "tuned_recall": tR,
            "n_pairs": n_pairs, "n_gold": n_gold, "prevalence": n_gold / max(n_pairs, 1),
            "cov_pairs": seen_pairs / max(n_pairs, 1), "cov_gold": seen_gold / max(n_gold, 1)}


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--repeats", type=int, default=10, help="independent fold shufflings")
    ap_.add_argument("--alpha", type=float, default=5.0)
    ap_.add_argument("--scope", choices=("intra", "inter", "all"), default="intra")
    ap_.add_argument("--json-out", type=Path, default=None)
    args = ap_.parse_args()

    cache: Dict[str, Tuple[List[List[Pair]], List[str], List[str]]] = {}
    out: Dict[str, dict] = {}

    print(f"scope: {args.scope}-sentence pairs")
    for ds_key, label, n_folds, by_topic in PROTOCOLS:
        if ds_key not in cache:
            docs = list(load_hf_dataset_parsed(DATASETS[ds_key], split="train"))
            lemmatize(docs)
            kept = [(doc_pairs_scoped(d, args.scope), d["id"]) for d in docs]
            kept = [(p, i) for p, i in kept if p]
            cache[ds_key] = ([p for p, _ in kept],
                             [i for _, i in kept],
                             [group_of(i) for _, i in kept])
        per_doc, ids, topics = cache[ds_key]
        units = topics if by_topic else ids

        runs = [cross_validate(per_doc, units, n_folds, args.alpha, seed)
                for seed in range(args.repeats)]
        agg = {k: (statistics.mean([r[k] for r in runs]),
                   statistics.pstdev([r[k] for r in runs])) for k in runs[0]}
        out[label] = {"runs": runs, "mean_sd": agg,
                      "n_docs": len(per_doc), "n_units": len(set(units)), "n_folds": n_folds}

        print(f"\n{label}")
        print(f"  {len(per_doc)} docs, {len(set(units))} split units, {n_folds} folds, "
              f"{args.repeats} shufflings")
        print(f"  {agg['n_pairs'][0]:.0f} intra pairs, {agg['n_gold'][0]:.0f} positives "
              f"({agg['prevalence'][0]:.1%} prevalence)")
        for name, key in (("average precision", "ap"),
                          ("F1 (nested threshold)", "tuned_f1"),
                          ("   precision", "tuned_precision"),
                          ("   recall", "tuned_recall"),
                          ("F1 (oracle threshold)", "f1"),
                          ("   precision", "precision"), ("   recall", "recall"),
                          ("pairs with lemma pair seen", "cov_pairs"),
                          ("positives with lemma pair seen", "cov_gold")):
            m, sd = agg[key]
            print(f"    {name:<32}{m:.3f} ±{sd:.3f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
