"""Trigger-lemma-pair prior as a standalone predictor, on an 80/20 document split.

Question this answers: how much of the causal annotation is recoverable from nothing but
"which pair of trigger lemmas is this", with no document, no model and no reasoning? The
point is not to beat the LLM but to size a cheap prior that could gate it — in particular
its use as a *veto* (lemma pairs the annotators never link), which is the arm that fits a
precision-first cascade.

Scoring is a backoff chain with additive smoothing, each level shrunk toward the one below:
    global -> source/target lemma marginals (geometric mean) -> undirected pair -> directed pair
Every level is estimated on the training documents only, so no test document contributes
counts to its own prediction.

Two split granularities are reported, because they disagree wildly on EventStoryLine. ESC
documents come from ECB+ topics — ~11 documents per topic, all reporting the *same* real-world
event, so they share triggers and share annotated relations. A random document split therefore
trains on near-duplicates of its own test documents; the topic split (the standard ESC protocol)
holds out whole topics and is the number to believe. Causal-TimeBank has no topic grouping —
its ids are TimeBank filenames — so both splits are the same thing there.

    uv run python scripts/experiments/lexical_prior.py
    uv run python scripts/experiments/lexical_prior.py --seeds 10 --alpha 3 --all-pairs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataprep.dataprep import load_hf_dataset_parsed  # noqa: E402
from utils.syntax import _build_doc, _load_pipeline, _span_root  # noqa: E402

DATASETS = {
    "causal_timebank": "Nofing/CausalTimeBank-standard-eci",
    "event_story_line": "Nofing/EventStoryLine-0.9-standard-eci",
}

Pair = Tuple[str, str, bool]  # (source lemma, target lemma, is_gold_positive)


ECB_ID = re.compile(r"^(\d+)_\d+ecb", re.I)


def group_of(doc_id: str) -> str:
    """ECB+ topic for EventStoryLine (`1_10ecbplus.xml` -> `1`), the document itself otherwise.

    Deliberately narrow: Causal-TimeBank ids are TimeBank filenames, and a looser rule would
    fold every `wsj_*` document into one bogus "topic".
    """
    m = ECB_ID.match(doc_id)
    return m.group(1) if m else doc_id


# ── data ──────────────────────────────────────────────────────────────────────

def lemmatize(docs: List[dict]) -> None:
    """Attach `lemmas`: mention id -> lemma of the span's head token, lowercased.

    Uses the same Doc construction as utils/syntax.py, so the lemma is the one the model
    already sees in a `syntax.level: mentions` block: the dataset's own tokens and sentence
    boundaries, head found set-wise to survive discontinuous ESC spans.
    """
    nlp = _load_pipeline()
    if nlp is None:
        raise SystemExit("spaCy or en_core_web_sm is unavailable — no lemmas, no prior.")
    for doc in docs:
        tokens, spans = doc.get("tokens") or [], doc.get("spans") or []
        lemmas: Dict[str, str] = {}
        if tokens and spans:
            sdoc = _build_doc(nlp, tokens, doc["sentences"])
            for mid, span in zip(doc.get("mentions") or [], spans):
                idxs = [int(i) for i in span if 0 <= int(i) < len(sdoc)]
                if idxs:
                    lemmas[mid] = _span_root(sdoc, idxs).lemma_.lower()
        doc["lemmas"] = lemmas


def doc_pairs(doc: dict, intra_only: bool) -> List[Pair]:
    """Candidate ordered pairs of one document as (source lemma, target lemma, gold)."""
    lemmas, sent = doc["lemmas"], doc["mention_sentence"]
    gold = {(s, t) for s, _lab, t in doc["gold_triples"]}
    out: List[Pair] = []
    for a, b in doc["pair_list_ids"]:
        if a not in lemmas or b not in lemmas:
            continue
        if intra_only and sent.get(a, -1) != sent.get(b, -2):
            continue
        out.append((lemmas[a], lemmas[b], (a, b) in gold))
    return out


# ── model ─────────────────────────────────────────────────────────────────────

class LemmaPrior:
    """Counts over the training documents, plus the smoothed backoff score."""

    def __init__(self, train: Sequence[List[Pair]], alpha: float):
        self.alpha = alpha
        self.ab: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
        self.un: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
        self.src: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.tgt: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        self.tot = [0, 0]
        for pairs in train:
            for a, b, y in pairs:
                for bucket in (self.ab[(a, b)], self.un[tuple(sorted((a, b)))],
                               self.src[a], self.tgt[b], self.tot):
                    bucket[0] += 1
                    bucket[1] += int(y)

    @staticmethod
    def _shrink(bucket: Sequence[int], prior: float, alpha: float) -> float:
        n, pos = bucket
        return (pos + alpha * prior) / (n + alpha)

    def score(self, a: str, b: str) -> float:
        g = self.tot[1] / max(self.tot[0], 1)
        s = self._shrink(self.src.get(a, (0, 0)), g, self.alpha)
        t = self._shrink(self.tgt.get(b, (0, 0)), g, self.alpha)
        marg = math.sqrt(max(s, 1e-9) * max(t, 1e-9))
        u = self._shrink(self.un.get(tuple(sorted((a, b))), (0, 0)), marg, self.alpha)
        return self._shrink(self.ab.get((a, b), (0, 0)), u, self.alpha)

    def seen(self, a: str, b: str) -> Tuple[int, int]:
        return tuple(self.ab.get((a, b), (0, 0)))  # type: ignore[return-value]


# ── metrics ───────────────────────────────────────────────────────────────────

def pr_curve(scored: List[Tuple[float, bool]]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Average precision, and the (recall, precision, threshold) points of the sweep."""
    scored = sorted(scored, key=lambda x: -x[0])
    npos = sum(y for _s, y in scored) or 1
    tp = fp = 0
    ap, prev_r = 0.0, 0.0
    pts: List[Tuple[float, float, float]] = []
    for i, (s, y) in enumerate(scored):
        tp, fp = tp + int(y), fp + int(not y)
        if i + 1 < len(scored) and scored[i + 1][0] == s:
            continue
        p, r = tp / (tp + fp), tp / npos
        ap += p * (r - prev_r)
        prev_r = r
        pts.append((r, p, s))
    return ap, pts


def at_recall(pts: Sequence[Tuple[float, float, float]], target: float) -> float:
    """Best precision reachable at recall >= target (0.0 if unreachable)."""
    return max((p for r, p, _s in pts if r >= target), default=0.0)


def best_f1(pts: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    for r, p, _s in pts:
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best[0]:
            best = (f, p, r)
    return best


def mean_sd(xs: Sequence[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


# ── one split ─────────────────────────────────────────────────────────────────

def run_split(per_doc: List[List[Pair]], groups: Sequence[str], seed: int,
              alpha: float, train_frac: float) -> dict:
    """One 80/20 split. Documents are held out in whole `groups`, so a group never straddles
    the split — with one group per document this is a plain document split."""
    keys = sorted(set(groups))
    random.Random(seed).shuffle(keys)
    cut = int(round(train_frac * len(keys)))
    train_keys = set(keys[:cut])
    train = [p for p, g in zip(per_doc, groups) if g in train_keys]
    test = [p for p, g in zip(per_doc, groups) if g not in train_keys]
    prior = LemmaPrior(train, alpha)

    test_pairs = [p for doc in test for p in doc]
    scored = [(prior.score(a, b), y) for a, b, y in test_pairs]
    ap, pts = pr_curve(scored)
    f1, p_at_f1, r_at_f1 = best_f1(pts)
    npos = sum(y for _a, _b, y in test_pairs) or 1

    # The honest counterpart to best_f1: counts from the first half of the training groups,
    # threshold chosen on the second half, both applied to the untouched test groups. best_f1
    # peeks at the test labels to place its threshold, so it is an upper bound no deployable
    # decision rule reaches; the gap between the two is that optimism, measured.
    inner = keys[:cut]
    icut = max(1, int(round(0.75 * len(inner))))
    fit_keys, dev_keys = set(inner[:icut]), set(inner[icut:])
    fit = [p for p, g in zip(per_doc, groups) if g in fit_keys]
    dev_pairs = [x for p, g in zip(per_doc, groups) if g in dev_keys for x in p]
    tuned_f1 = tuned_p = tuned_r = 0.0
    if fit and dev_pairs:
        inner_prior = LemmaPrior(fit, alpha)
        _ap, dev_pts = pr_curve([(inner_prior.score(a, b), y) for a, b, y in dev_pairs])
        thr = max(dev_pts, key=lambda x: (2 * x[1] * x[0] / (x[1] + x[0])) if x[0] + x[1] else 0.0)[2]
        tp = sum(1 for s_, y in scored if s_ >= thr and y)
        fp = sum(1 for s_, y in scored if s_ >= thr and not y)
        tuned_p = tp / (tp + fp) if tp + fp else 0.0
        tuned_r = tp / npos
        tuned_f1 = 2 * tuned_p * tuned_r / (tuned_p + tuned_r) if tuned_p + tuned_r else 0.0

    # Coverage and the two ways the prior can be used as a gate.
    seen_pos = seen_any = seen3 = 0
    ever_tp = ever_fp = 0            # positive gate: lemma pair was linked at least once in train
    veto_hit = veto_wrong = 0        # negative gate: pair seen >= 3 times, never linked
    for a, b, y in test_pairs:
        n, pos = prior.seen(a, b)
        seen_any += n >= 1
        seen3 += n >= 3
        seen_pos += (n >= 1) and y
        if pos >= 1:
            ever_tp += y
            ever_fp += not y
        if n >= 3 and pos == 0:
            veto_hit += 1
            veto_wrong += y

    return {
        "n_train_groups": float(len(train_keys)), "n_test_groups": float(len(keys) - len(train_keys)),
        "n_test_docs": len(test), "n_test_pairs": len(test_pairs), "n_test_pos": npos,
        "prevalence": npos / max(len(test_pairs), 1),
        "ap": ap, "best_f1": f1, "p_at_best_f1": p_at_f1, "r_at_best_f1": r_at_f1,
        "tuned_f1": tuned_f1, "p_at_tuned": tuned_p, "r_at_tuned": tuned_r,
        "p_at_r20": at_recall(pts, 0.20), "p_at_r40": at_recall(pts, 0.40),
        "p_at_r55": at_recall(pts, 0.55), "p_at_r70": at_recall(pts, 0.70),
        "cov_pairs": seen_any / max(len(test_pairs), 1),
        "cov_pairs_n3": seen3 / max(len(test_pairs), 1),
        "cov_gold": seen_pos / npos,
        "ever_p": ever_tp / max(ever_tp + ever_fp, 1),
        "ever_r": ever_tp / npos,
        "veto_frac": veto_hit / max(len(test_pairs), 1),
        "veto_recall_cost": veto_wrong / npos,
        "veto_purity": 1 - veto_wrong / max(veto_hit, 1),
    }


# ── cascade: the veto applied to a real LLM run ───────────────────────────────

def cascade_on_llm_run(run_path: Path, docs: List[dict], intra_only: bool,
                       alpha: float, n_folds: int = 5) -> None:
    """What the negative gate would do to an already-logged LLM run.

    Cross-topic by construction: the prior scoring a document's pairs is counted only from
    documents in other ECB+ topics, so nothing a near-duplicate document taught it can leak
    back in. The veto can only flip a predicted positive to negative, so it trades false
    positives against true ones and never invents a relation.
    """
    rows = [r for r in json.load(run_path.open())["results"]["per_pair_predictions"]
            if r.get("sentence_relation") == "intra"]
    by_id = {d["id"]: d for d in docs}
    rows = [r for r in rows if r["id"] in by_id]
    topics = sorted({group_of(d["id"]) for d in docs})
    fold_of = {t: i % n_folds for i, t in enumerate(topics)}

    counts: Dict[Tuple[str, str], List[int]] = {}
    for fold in range(n_folds):
        train = [doc_pairs(d, intra_only) for d in docs if fold_of[group_of(d["id"])] != fold]
        prior = LemmaPrior(train, alpha)
        for r in rows:
            if fold_of[group_of(r["id"])] != fold:
                continue
            doc = by_id[r["id"]]
            a, _, b = r["pair"].partition(",")
            la, lb = doc["lemmas"].get(a), doc["lemmas"].get(b)
            counts[(r["id"], r["pair"])] = list(prior.seen(la, lb)) if la and lb else [0, 0]

    def pos(label) -> bool:
        return label not in ("norel", None, "")

    gold_pos = sum(pos(r["gold"]) for r in rows)
    print(f"\n{'=' * 86}\ncascade — lemma-pair veto over {run_path.name}"
          f"\n{'=' * 86}")
    print(f"  {len(rows)} intra rows, {gold_pos} gold positives, "
          f"{n_folds}-fold cross-topic prior")
    print(f"\n  {'veto rule':<26}{'FPs cut':>9}{'TPs cut':>9}{'P':>8}{'R':>8}{'F1':>8}")
    for m in (0, 1, 2, 3, 5):
        tp = fp = cut_fp = cut_tp = 0
        for r in rows:
            if not pos(r["pred"]):
                continue
            n, npos_ = counts.get((r["id"], r["pair"]), [0, 0])
            vetoed = m > 0 and n >= m and npos_ == 0
            if vetoed:
                cut_tp += pos(r["gold"])
                cut_fp += not pos(r["gold"])
                continue
            tp += pos(r["gold"])
            fp += not pos(r["gold"])
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / gold_pos if gold_pos else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        label = "none (run as logged)" if m == 0 else f"seen >= {m}x, never linked"
        print(f"  {label:<26}{cut_fp:>9}{cut_tp:>9}{P:>8.3f}{R:>8.3f}{F:>8.3f}")


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--seeds", type=int, default=5)
    ap_.add_argument("--alpha", type=float, default=5.0)
    ap_.add_argument("--train-frac", type=float, default=0.8)
    ap_.add_argument("--all-pairs", action="store_true",
                     help="score every candidate pair, not just intra-sentence ones")
    ap_.add_argument("--llm-run", type=Path, default=None,
                     help="an EventStoryLine run_*.json; reports what the veto does to it")
    ap_.add_argument("--json-out", type=Path, default=None)
    args = ap_.parse_args()

    intra_only = not args.all_pairs
    scope = "all candidate pairs" if args.all_pairs else "intra-sentence pairs only"
    report: Dict[str, dict] = {}

    for name, repo in DATASETS.items():
        docs = list(load_hf_dataset_parsed(repo, split="train"))
        lemmatize(docs)
        kept = [(doc_pairs(d, intra_only), d["id"]) for d in docs]
        kept = [(p, i) for p, i in kept if p]
        per_doc = [p for p, _i in kept]
        by_doc = [i for _p, i in kept]
        by_topic = [group_of(i) for _p, i in kept]

        modes = {"document": by_doc}
        if len(set(by_topic)) < len(set(by_doc)):
            modes["topic"] = by_topic

        print(f"\n{'=' * 86}\n{name}  —  {scope}, "
              f"{int(args.train_frac * 100)}/{100 - int(args.train_frac * 100)} split, "
              f"{args.seeds} seeds, alpha={args.alpha}\n{'=' * 86}")

        aggs = {}
        for mode, groups in modes.items():
            splits = [run_split(per_doc, groups, seed, args.alpha, args.train_frac)
                      for seed in range(args.seeds)]
            aggs[mode] = {k: mean_sd([s[k] for s in splits]) for k in splits[0]}
            report.setdefault(name, {})[mode] = {"splits": splits, "mean_sd": aggs[mode]}

        first = next(iter(aggs.values()))
        print(f"  docs {len(per_doc)}   held-out units per split: "
              + "   ".join(f"{m} {a['n_test_groups'][0]:.0f}/{a['n_train_groups'][0] + a['n_test_groups'][0]:.0f}"
                           for m, a in aggs.items())
              + f"\n  test pairs ~{first['n_test_pairs'][0]:.0f}"
              f"   gold positives ~{first['n_test_pos'][0]:.0f}"
              f"   prevalence {first['prevalence'][0]:.3f}")
        head = "".join(f"{m + ' split':>20}" for m in aggs)
        print(f"\n  {'metric':<34}{head}")
        rows = [
            ("average precision", "ap"),
            ("best F1 (oracle threshold)", "best_f1"),
            ("   its precision", "p_at_best_f1"),
            ("   its recall", "r_at_best_f1"),
            ("F1 (threshold tuned on dev)", "tuned_f1"),
            ("   its precision", "p_at_tuned"),
            ("   its recall", "r_at_tuned"),
            ("precision @ recall 0.20", "p_at_r20"),
            ("precision @ recall 0.40", "p_at_r40"),
            ("precision @ recall 0.55", "p_at_r55"),
            ("precision @ recall 0.70", "p_at_r70"),
            ("test pairs w/ lemma pair seen", "cov_pairs"),
            ("   ... seen >= 3 times", "cov_pairs_n3"),
            ("gold positives w/ pair seen", "cov_gold"),
            ("memorise arm: precision", "ever_p"),
            ("memorise arm: recall", "ever_r"),
            ("veto arm: pairs vetoed", "veto_frac"),
            ("veto arm: gold positives lost", "veto_recall_cost"),
            ("veto arm: purity", "veto_purity"),
        ]
        for label, key in rows:
            cells = "".join(f"{aggs[m][key][0]:>13.3f} ±{aggs[m][key][1]:<5.3f}" for m in aggs)
            print(f"  {label:<34}{cells}")

        if args.llm_run and name == "event_story_line":
            cascade_on_llm_run(args.llm_run, docs, intra_only, args.alpha)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
