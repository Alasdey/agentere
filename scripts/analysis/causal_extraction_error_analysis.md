# Do different models make the same mistakes? An intra-sentence error-agreement analysis of the `agentere` MLflow runs

## Summary

This report analyzes every intra-sentence event-pair prediction logged in `agentere`'s `mlflow.db` that was scored by more than five qualifying runs, to test whether different language models make the same causal-extraction errors or fail independently of one another.

The data says the errors are shared, not independent. Across 34 qualifying runs spanning 8 models, predictions on the same recurring set of intra-sentence pairs agree far more than chance would allow (Fleiss' κ = 0.58 for the 3-way label, 0.70 for the binary causal/not-causal framing). 14.4% of the repeatedly-scored samples get a literally unanimous prediction from every run that touched them, and 95.7% of those unanimous verdicts are wrong. When models agree and are wrong, they agree on a specific *kind* of wrong: 93% of unanimous errors are "phantom causality" — inventing a causal relation where the annotation says there is none. Every pair of models shows a positive correlation between their correctness on shared samples (φ from 0.31 to 0.65), and no single model breaks away from the pack — all 8 models score within a 19–29% accuracy band on this hard, recurring slice of the data.

## 1. Data and methodology

**Source.** `/home/jovyan/project/temp/agentere/mlflow.db`, an MLflow SQLite backend, plus the run artifact JSON files under `mlruns/`. The task is event causality identification: for a candidate pair of event/entity mentions in a document, a model predicts one of three labels — `causes`, `causedby`, or `norel` (no relation) — against a human-annotated gold label.

**Run selection.** The database contains 240 logged runs. Filtering to runs with a logged `total_pairs > 10` and `binary_intra_f1 > 0.30` yields **34 qualifying runs**. These span 8 distinct models and two datasets:

| Dataset | Split | Qualifying runs |
|---|---|---|
| MAVEN-ERE-Causal | test | 30 |
| EventStoryLine-0.9 | train | 4 |

| Model | Qualifying runs |
|---|---|
| DeepSeek-V3.2 | 12 |
| DeepSeek-V3.2 (nitro) | 9 |
| DeepSeek-V4-Pro | 5 |
| GPT-5.5 | 3 |
| Gemini-2.5-Flash-Lite | 2 |
| GLM-5.2 | 1 |
| Qwen3-30B-A3B-Thinking | 1 |
| Gemini-3.1-Flash-Lite | 1 |

**Sample definition.** Each run's artifact JSON logs a `per_pair_predictions` list. Each entry is keyed by `(dataset, document id, entity pair)` — for example `pair: "e11,e9"` — and carries a gold label, a predicted label, and a `sentence_relation` tag (`intra`, `inter`, or `unknown`) that records whether the two mentions in the pair sit in the same sentence or different ones. A "sample" in this report is one such `(dataset, doc_id, pair)` key.

**Scope: intra-sentence only.** This analysis keeps only rows with `sentence_relation == "intra"` — the same scope the `binary_intra_f1` metric (used above to qualify runs) is computed on. Across the 34 qualifying runs there are 46,316 intra-sentence pair-observations (and 93,235 inter-sentence ones, excluded here). These 46,316 observations cover **19,948 distinct intra-sentence samples**. Restricting further to samples scored by more than 5 of the 34 qualifying runs — the population this report analyzes — leaves **1,294 samples (6.5% of all distinct intra-sentence samples)**. All 1,294 belong to MAVEN-ERE-Causal; EventStoryLine-0.9 only has 4 qualifying runs total, so no EventStoryLine sample can reach the >5 threshold regardless of scope.

Coverage per sample is uneven, because different runs used different `max_examples`, resampling, and few-shot settings: a given sample can be scored by anywhere from 6 to 30 of the 34 qualifying runs (median 18). The full coverage histogram is in the appendix.

Gold labels never disagreed across runs for a given sample — a basic data-integrity check that passed cleanly — so "gold" is unambiguous throughout.

**A note on a computational pitfall avoided here.** Measuring whether two runs are "wrong together" more often than chance requires care with how boolean correctness values are combined after a pivot/reshape operation: if a boolean column picks up missing values and gets silently cast to a generic object type, Python's bitwise-NOT operator on a plain `bool` does not return the logical negation (`~True` is `-2`, not `False`) — it will make every row look like a shared failure regardless of the actual data. All boolean logic in this analysis explicitly casts to true boolean type before any bitwise operations, and the co-occurrence figures below were verified against an independent Pearson-correlation formulation (φ, see §6) that does not rely on bitwise inversion at all.

## 2. Headline numbers

| Metric | Value |
|---|---|
| Qualifying runs / models | 34 / 8 |
| Distinct intra-sentence samples (any coverage) | 19,948 |
| Samples scored by >5 qualifying runs (this report's population) | 1,294 (6.5%) |
| Fleiss' κ, 3-way label (causes / causedby / norel) | **0.58** |
| Fleiss' κ, binary (causal vs. norel) | **0.70** |
| Samples with a unanimous prediction across all covering runs | 186 / 1,294 (14.4%) |
| ...of which unanimously wrong | 178 / 186 (95.7%) |
| ...of which unanimously correct | 8 / 186 (4.3%) |
| Majority-vote (ensemble) accuracy on this sample set | 16.2% |
| Mean pairwise φ (correctness correlation), all 435 run pairs | **0.45** |
| Mean pairwise raw prediction agreement | 69.1% |
| Mean pairwise Cohen's κ | 0.53 |

Both Fleiss' κ values sit in the "moderate-to-substantial" agreement band (Landis–Koch scale) — well above what independent, uncorrelated model behavior would produce (κ near 0), but not so high that every model behaves identically (κ near 1).

## 3. Models converge on the same verdict — usually the wrong one

The per-sample error rate — the fraction of covering runs that got a given sample wrong — is bimodal and skews heavily toward "everyone wrong":

| Error rate across covering runs | Samples |
|---|---|
| 0 (every run correct) | 8 |
| (0, 0.2] | 62 |
| (0.2, 0.4] | 88 |
| (0.4, 0.6] | 84 |
| (0.6, 0.8] | 110 |
| (0.8, 1) | 146 |
| 1 (every run wrong) | 796 |

Nearly 62% of the 1,294 repeatedly-scored samples (796) are wrong in *every single run* that touched them, regardless of which model or configuration was used. Only 8 samples are correct in every covering run.

Correlation between per-sample prediction entropy (how spread out the runs' predicted labels are) and per-sample error rate is −0.42: the lowest-entropy samples (where runs agree most tightly) are disproportionately the ones with the highest error rates. Agreement and correctness are pulling in opposite directions here — models aren't converging on truth, they're converging on a shared blind spot.

## 4. The shared mistake has a specific shape

Restricting to the 178 samples where every covering run unanimously agreed on a wrong label, and cross-tabulating gold label against the (unanimous) predicted label:

| Gold label | Unanimous pred: causes | Unanimous pred: causedby | Unanimous pred: norel |
|---|---|---|---|
| causes | 0 | 0 | 83 |
| causedby | 0 | 0 | 83 |
| norel | 6 | 6 | 0 |

Two things stand out:

1. **When models agree and are wrong, they never confuse the two causal directions with each other.** No unanimous-wrong sample has gold `causes` and unanimous prediction `causedby`, or vice versa — direction-flipping essentially doesn't happen in the unanimous-agreement subset.
2. **The dominant failure mode is "phantom causality."** 166 of the 178 unanimous-wrong samples (93%) have gold `norel`, with every model unanimously assigning a causal label instead — split almost exactly evenly between `causes` (83) and `causedby` (83). Genuinely missing a real intra-sentence causal relationship (gold `causes`/`causedby`, unanimous prediction `norel`) accounts for only 12 of the 178 cases (7%).

Put plainly: for intra-sentence pairs, when every model agrees and is wrong, it is almost always because every model saw a causal relationship that the annotators say is not there, not because every model missed one that is.

## 5. No model breaks away from the pack

Binary (causal vs. `norel`) accuracy on the 1,294-sample repeated subset, by model:

| Model | Binary accuracy | 3-way exact-match accuracy | Observations |
|---|---|---|---|
| Gemini-2.5-Flash-Lite | 29.3% | 28.6% | 1,196 |
| DeepSeek-V3.2 | 28.9% | 23.2% | 10,281 |
| DeepSeek-V4-Pro | 27.8% | 27.7% | 1,970 |
| GPT-5.5 | 27.8% | 27.8% | 1,058 |
| GLM-5.2 | 25.6% | 25.6% | 782 |
| Qwen3-30B-A3B-Thinking | 25.2% | 25.2% | 882 |
| DeepSeek-V3.2 (nitro) | 21.6% | 21.3% | 5,625 |
| Gemini-3.1-Flash-Lite | 18.7% | 17.9% | 780 |

All 8 models fall within a single 19–29 percentage-point band. There is no model here that is dramatically better or worse at this specific, recurring, hard slice of the task — the spread (10.5 points top-to-bottom) is small next to how far every model is from ceiling performance. Notably, DeepSeek-V3.2 and its "nitro" serving-tier variant — nominally the same underlying model — differ by 7.3 points, a reminder that serving configuration, not just base model identity, moves these numbers.

## 6. Pairwise agreement, correlation, and shared error — every model pair

For every pair of the 34 qualifying runs, restricted to the intra-sentence samples both runs covered, this analysis computes:

- **Raw agreement** — do the two runs predict the same label on the same samples?
- **Cohen's κ** — raw agreement corrected for the agreement expected by chance given each run's label distribution.
- **φ (phi correlation of correctness)** — the Pearson correlation between "run i got sample X right" and "run j got sample X right," treated as two binary variables over the shared samples. φ = 0 means the two runs' correctness is statistically independent of each other; φ > 0 means they tend to be right, or wrong, together on the same items.
- **Co-error lift** — P(both wrong) ÷ [P(run i wrong) × P(run j wrong)]. A value of 1.0 means the two runs' errors co-occur exactly as often as independence would predict; higher means their errors cluster together more than chance. (This measure is included for completeness but is naturally compressed toward 1.0 here because individual error rates are already high — 60–80% for most runs on this subset — which mechanically limits how much lift a correlation can produce. φ is the more informative of the two summary statistics for this reason.)

Aggregated across all 435 run pairs (weighted by the number of shared samples):

| Statistic | Weighted mean | Range across all 435 pairs |
|---|---|---|
| Raw agreement | 69.1% | 18.9% – 97.1% |
| Cohen's κ | 0.53 | −0.16 – 0.96 |
| Co-error lift | 1.19× | 0.95× – 1.47× |
| φ (correctness correlation) | 0.45 | −0.23 – 0.93 |

Rolling run pairs up to the 28 cross-model pairs (i.e. excluding pairs that are two different runs of the *same* model), sorted by φ:

| Model A | Model B | Shared samples | Raw agreement | Cohen's κ | Co-error lift | φ |
|---|---|---|---|---|---|---|
| GPT-5.5 | GLM-5.2 | 892 | 84.8% | 0.76 | 1.28× | 0.65 |
| DeepSeek-V4-Pro | Qwen3-30B-A3B-Thinking | 1,614 | 82.0% | 0.73 | 1.28× | 0.59 |
| DeepSeek-V4-Pro | GPT-5.5 | 2,236 | 81.8% | 0.72 | 1.28× | 0.59 |
| DeepSeek-V4-Pro | GLM-5.2 | 1,560 | 80.1% | 0.69 | 1.25× | 0.55 |
| DeepSeek-V3.2 (nitro) | GLM-5.2 | 4,630 | 78.7% | 0.65 | 1.20× | 0.51 |
| DeepSeek-V3.2 | GLM-5.2 | 7,332 | 69.7% | 0.55 | 1.22× | 0.50 |
| Qwen3-30B-A3B-Thinking | GLM-5.2 | 672 | 77.7% | 0.64 | 1.22× | 0.48 |
| Gemini-2.5-Flash-Lite | Qwen3-30B-A3B-Thinking | 874 | 75.1% | 0.63 | 1.24× | 0.48 |
| GPT-5.5 | Qwen3-30B-A3B-Thinking | 886 | 77.0% | 0.64 | 1.23× | 0.48 |
| DeepSeek-V3.2 | Qwen3-30B-A3B-Thinking | 7,874 | 69.0% | 0.54 | 1.21× | 0.47 |
| DeepSeek-V4-Pro | Gemini-2.5-Flash-Lite | 1,990 | 74.0% | 0.61 | 1.24× | 0.47 |
| DeepSeek-V3.2 (nitro) | Qwen3-30B-A3B-Thinking | 4,680 | 76.7% | 0.62 | 1.19× | 0.47 |
| DeepSeek-V3.2 | DeepSeek-V4-Pro | 17,682 | 68.5% | 0.52 | 1.21× | 0.47 |
| DeepSeek-V3.2 (nitro) | DeepSeek-V4-Pro | 10,596 | 75.0% | 0.61 | 1.20× | 0.46 |
| DeepSeek-V3.2 | Gemini-2.5-Flash-Lite | 10,058 | 65.9% | 0.47 | 1.21× | 0.46 |
| DeepSeek-V3.2 (nitro) | Gemini-2.5-Flash-Lite | 5,732 | 71.1% | 0.57 | 1.20× | 0.45 |
| Gemini-2.5-Flash-Lite | GLM-5.2 | 798 | 70.4% | 0.56 | 1.23× | 0.44 |
| DeepSeek-V3.2 | GPT-5.5 | 9,992 | 66.9% | 0.50 | 1.19× | 0.43 |
| Gemini-3.1-Flash-Lite | GLM-5.2 | 676 | 75.7% | 0.59 | 1.13× | 0.40 |
| DeepSeek-V3.2 (nitro) | Gemini-3.1-Flash-Lite | 4,650 | 77.8% | 0.61 | 1.12× | 0.40 |
| DeepSeek-V3.2 | DeepSeek-V3.2 (nitro) | 51,330 | 65.4% | 0.48 | 1.16× | 0.40 |
| Gemini-3.1-Flash-Lite | Qwen3-30B-A3B-Thinking | 672 | 73.2% | 0.56 | 1.14× | 0.39 |
| DeepSeek-V4-Pro | Gemini-3.1-Flash-Lite | 1,500 | 71.1% | 0.54 | 1.15× | 0.37 |
| DeepSeek-V3.2 (nitro) | GPT-5.5 | 5,994 | 72.5% | 0.56 | 1.16× | 0.37 |
| DeepSeek-V3.2 | Gemini-3.1-Flash-Lite | 7,291 | 63.0% | 0.45 | 1.12× | 0.34 |
| Gemini-3.1-Flash-Lite | GPT-5.5 | 854 | 70.5% | 0.52 | 1.12× | 0.32 |
| Gemini-2.5-Flash-Lite | Gemini-3.1-Flash-Lite | 782 | 65.2% | 0.49 | 1.14× | 0.31 |
| Gemini-2.5-Flash-Lite | GPT-5.5 | 1,152 | 65.6% | 0.49 | 1.17× | 0.31 |

**Every one of these 28 cross-model pairs has φ > 0.** There is no pair of models in this dataset whose correctness on shared intra-sentence samples behaves as if independent — the weakest correlation (Gemini-2.5-Flash-Lite × GPT-5.5, φ = 0.31) is still a moderate positive relationship, and the strongest (GPT-5.5 × GLM-5.2, φ = 0.65) is a strong one.

One run stands out as an outlier against the rest of the field: a GPT-5.5 run configured with few-shot prompting disabled and only 128 total pairs produces the lowest — and only mildly negative — φ values in the dataset (as low as −0.23 against a DeepSeek-V3.2-nitro run, on just 58–68 shared samples). Given the very small sample overlap and the atypical configuration (no few-shot examples), this reads as a noisy, under-powered comparison rather than evidence that models can diverge — every other pair, including every other pair involving GPT-5.5, is positively correlated.

Same-model comparisons (different runs/configurations of one model, compared to each other) are also informative:

| Model | Run pairs compared | Shared samples | Raw agreement | Cohen's κ | Co-error lift | φ |
|---|---|---|---|---|---|---|
| GPT-5.5 | 3 | 280 | 92.9% | 0.89 | 1.38× | 0.84 |
| Gemini-2.5-Flash-Lite | 1 | 80 | 82.5% | 0.73 | 1.28× | 0.75 |
| DeepSeek-V4-Pro | 3 | 1,066 | 85.6% | 0.78 | 1.33× | 0.67 |
| DeepSeek-V3.2 (nitro) | 21 | 14,527 | 83.5% | 0.73 | 1.19× | 0.58 |
| DeepSeek-V3.2 | 66 | 40,102 | 62.6% | 0.43 | 1.19× | 0.47 |

Same-model self-consistency (φ 0.47–0.84) is generally higher than cross-model correlation (φ 0.31–0.65), as expected, but the gap is not large — DeepSeek-V3.2's own self-consistency (φ = 0.47, driven by 66 run pairs that differ in few-shot settings, resampling, and tool use) is within the range of several cross-model pairs. Prompting/configuration choices move a model's behavior on this hard sample set almost as much as swapping in a different model entirely.

## 7. Is "scored by >5 runs" the same as "hard"?

Samples that recur across more than 5 qualifying runs are not simply an arbitrary or artificially-hardened subset — but they are not neutral either. Comparing the 1,294 repeated intra-sentence samples against the 18,654 intra-sentence samples seen in 5 or fewer qualifying runs:

| | Repeated (>5 runs) | Non-repeated (≤5 runs) |
|---|---|---|
| Binary accuracy | 26.3% | 22.1% |
| 3-way accuracy | 23.6% | 16.7% |
| Gold label share: causal (`causes`+`causedby`) | 59.7% | 47.2% |
| Gold label share: `norel` | 40.3% | 52.8% |

Repeated samples score *somewhat higher* on both accuracy measures than non-repeated ones — so recurrence across runs is not a proxy for "the hardest possible examples." What does shift is the label mix: repeated samples are moderately enriched for genuine causal relationships (59.7% vs. 47.2% base rate) relative to `norel` pairs. This is consistent with true causal edges being fixed dataset annotations that tend to appear across most run configurations, while some `norel` candidate pairs are subject to more variable candidate-generation or subsampling behavior between runs.

Within the repeated-sample population itself, the 1,294-sample gold-label distribution is 21.1% `causes`, 21.1% `causedby`, 57.8% `norel` — close to, but not identical to, the 26.6% / 26.6% / 46.7% base rate across all intra-sentence pairs.

## 8. Interpretation

Four lines of evidence in this analysis point the same direction:

1. Fleiss' κ (0.58–0.70) shows model predictions on repeated samples are far more consistent with each other than random labeling would produce.
2. Nearly 62% of repeated samples are wrong in every single covering run, and when models unanimously agree, they are wrong 95.7% of the time.
3. The unanimous-wrong errors are not scattered noise — 93% of them are the same specific failure (inventing a causal relationship for a `norel` pair), not a grab-bag of different mistakes.
4. Every one of the 28 cross-model pairs shows positive correlation (φ 0.31–0.65) between which samples they get right and wrong, and per-model accuracy is compressed into a narrow 19–29% band with no standout performer.

Taken together, this is evidence of a shared, systematic weakness in how these models — across architectures, vendors, and serving configurations — handle a specific subset of intra-sentence causal-relation judgments, rather than each model having its own independent failure pattern. Ensembling by majority vote across models recovers only 16.2% accuracy on this sample set, barely above the best single model (Gemini-2.5-Flash-Lite, 29.3%) and well below what independent, uncorrelated errors would allow an ensemble to achieve — a direct, practical consequence of the errors being correlated rather than independent.

## Appendix

### A1. Per-sample run coverage

| Runs covering sample | Samples | Runs covering sample | Samples |
|---|---|---|---|
| 6 | 92 | 18 | 34 |
| 7 | 63 | 19 | 14 |
| 8 | 67 | 20 | 26 |
| 9 | 65 | 21 | 18 |
| 10 | 71 | 22 | 6 |
| 11 | 38 | 23 | 14 |
| 12 | 40 | 24 | 458 |
| 13 | 54 | 26 | 60 |
| 14 | 44 | 28 | 2 |
| 15 | 42 | 29 | 4 |
| 16 | 16 | 30 | 44 |
| 17 | 22 | | |

### A2. Data provenance

- Source: `/home/jovyan/project/temp/agentere/mlflow.db` (MLflow SQLite backend) and run artifacts under `mlruns/988445650945562991/<run_id>/artifacts/*.json`, read 2026-07-14.
- Run qualification query: joined the `binary_intra_f1` and `total_pairs` metrics from the MLflow `metrics` table, filtered to `total_pairs > 10 AND binary_intra_f1 > 0.30`.
- Per-sample predictions: `results.per_pair_predictions` in each qualifying run's logged JSON artifact, filtered to `sentence_relation == "intra"`.
- Analysis performed with pandas/numpy in the project's own `.venv` (pandas 2.3.3, numpy 2.4.3).
