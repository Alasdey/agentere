"""How much of the lookup table's score survives when the scoring is made naive?

Same folds, same counts, same protocols as lexical_prior_cv.py — only the function turning
counts into a score changes. All variants are evaluated in one pass over the folds, so any
difference between them is the scoring rule and nothing else.

    backoff   the smoothed chain: global -> lemma marginals -> undirected pair -> directed pair
    mle       raw proportion: positives / occurrences of that directed lemma pair; unseen -> 0
    mle>=3    the same proportion, but only where the pair occurs at least 3 times; else 0
    ever      no arithmetic at all: causal iff this lemma pair was ever a gold link in training

    uv run python scripts/experiments/lexical_prior_scorers.py --repeats 5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep.dataprep import load_hf_dataset_parsed  # noqa: E402
from scripts.experiments.lexical_prior import (  # noqa: E402
    DATASETS, LemmaPrior, Pair, best_f1, group_of, lemmatize, pr_curve,
)
from scripts.experiments.lexical_prior_cv import PROTOCOLS, doc_pairs_scoped, folds_of  # noqa: E402

Scorer = Callable[[LemmaPrior, str, str], float]


def _mle(prior: LemmaPrior, a: str, b: str) -> float:
    n, pos = prior.seen(a, b)
    return pos / n if n else 0.0


def _mle3(prior: LemmaPrior, a: str, b: str) -> float:
    n, pos = prior.seen(a, b)
    return pos / n if n >= 3 else 0.0


def _ever(prior: LemmaPrior, a: str, b: str) -> float:
    return 1.0 if prior.seen(a, b)[1] >= 1 else 0.0


SCORERS: Dict[str, Scorer] = {
    "backoff": lambda prior, a, b: prior.score(a, b),
    "mle": _mle,
    "mle>=3": _mle3,
    "ever": _ever,
}


def cross_validate(per_doc: Sequence[List[Pair]], units: Sequence[str],
                   n_folds: int, alpha: float, seed: int) -> Dict[str, dict]:
    """One CV pass, every scorer evaluated on exactly the same folds and counts."""
    pooled: Dict[str, List[Tuple[float, bool]]] = {k: [] for k in SCORERS}
    tuned: Dict[str, List[Tuple[bool, bool]]] = {k: [] for k in SCORERS}
    n_gold = 0

    for held in folds_of(units, n_folds, seed):
        tr_units = [u for u in sorted(set(units)) if u not in held]
        prior = LemmaPrior([p for p, u in zip(per_doc, units) if u not in held], alpha)

        inner_held = set(tr_units[::5])
        inner_prior = LemmaPrior([p for p, u in zip(per_doc, units)
                                  if u not in held and u not in inner_held], alpha)
        inner_pairs = [(a, b, y) for p, u in zip(per_doc, units) if u in inner_held
                       for a, b, y in p]

        thr: Dict[str, float] = {}
        for name, fn in SCORERS.items():
            curve = [(fn(inner_prior, a, b), y) for a, b, y in inner_pairs]
            if not curve:
                thr[name] = 0.0
                continue
            _ap, ipts = pr_curve(curve)
            thr[name] = max(ipts, key=lambda x: (2 * x[1] * x[0] / (x[1] + x[0]))
                            if x[0] + x[1] else 0.0)[2]

        for p, u in zip(per_doc, units):
            if u not in held:
                continue
            for a, b, y in p:
                n_gold += y
                for name, fn in SCORERS.items():
                    sc = fn(prior, a, b)
                    pooled[name].append((sc, y))
                    tuned[name].append((sc >= thr[name] and sc > 0.0, y))

    out = {}
    for name in SCORERS:
        ap, pts = pr_curve(pooled[name])
        f1, p, r = best_f1(pts)
        tp = sum(1 for pr_, y in tuned[name] if pr_ and y)
        fp = sum(1 for pr_, y in tuned[name] if pr_ and not y)
        tP = tp / (tp + fp) if tp + fp else 0.0
        tR = tp / max(n_gold, 1)
        out[name] = {"ap": ap, "oracle_f1": f1, "oracle_p": p, "oracle_r": r,
                     "f1": 2 * tP * tR / (tP + tR) if tP + tR else 0.0,
                     "precision": tP, "recall": tR}
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--repeats", type=int, default=5)
    ap_.add_argument("--alpha", type=float, default=5.0)
    ap_.add_argument("--json-out", type=Path, default=None)
    args = ap_.parse_args()

    cache: Dict[str, Tuple[List[List[Pair]], List[str], List[str]]] = {}
    report: Dict[str, dict] = {}

    for ds_key, label, n_folds, by_topic in PROTOCOLS:
        if ds_key not in cache:
            docs = list(load_hf_dataset_parsed(DATASETS[ds_key], split="train"))
            lemmatize(docs)
            kept = [(doc_pairs_scoped(d, "intra"), d["id"]) for d in docs]
            kept = [(p, i) for p, i in kept if p]
            cache[ds_key] = ([p for p, _ in kept], [i for _, i in kept],
                             [group_of(i) for _, i in kept])
        per_doc, ids, topics = cache[ds_key]
        units = topics if by_topic else ids

        runs = [cross_validate(per_doc, units, n_folds, args.alpha, s)
                for s in range(args.repeats)]
        print(f"\n{label}")
        print(f"  {'scorer':<10}{'AP':>16}{'F1':>16}{'precision':>16}{'recall':>16}")
        report[label] = {}
        for name in SCORERS:
            cells = []
            for key in ("ap", "f1", "precision", "recall"):
                vals = [r[name][key] for r in runs]
                m, sd = statistics.mean(vals), statistics.pstdev(vals)
                report[label].setdefault(name, {})[key] = [m, sd]
                cells.append(f"{m:.3f} ±{sd:.3f}")
            print(f"  {name:<10}" + "".join(f"{c:>16}" for c in cells))

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
