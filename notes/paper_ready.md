# Paper-ready results and analyses (as of 2026-07-19)

Only validated, citable content — organized by the paper section it feeds. Scores are
P / R / F1 ×100, binary causal identification, intra-sentence subset unless stated.
Items marked *(preliminary)* are directionally solid but rest on few seeds or an older
backbone; everything else is current-config and multiply checked.

## Main results (Table 1, SCOTECI row)

Full evaluation sets, deepseek-v4-pro, 3 demonstrations, 3-step synthetic CoT, N=3
resampling with majority vote, temperature 0:

| Dataset | Scope | P | R | F1 |
|---|---|---|---|---|
| Causal-TimeBank | all 183 docs | **61.1** | 39.6 | **48.1** |
| EventStoryLine 0.9 | 233 docs | 40.7 | 70.9 | **51.7** |
| MAVEN-ERE | 710 test docs (33k pairs) | **40.8** | 59.6 | 48.4 |

- **CTB**: +28.1 F1 over the best prior LLM method (SERE, 20.0); precision 61.1 where no
  LLM baseline exceeds 13.8. The degenerate high-recall/low-precision regime is broken on
  the dataset where positives are rarest.
- **ESC**: 51.7 vs SERE's 49.9.
- **MAVEN-ERE**: best precision in the table (40.8); F1 48.4 vs SERE's 42.3.
- Comparison framing: SERE evaluates intra-sentence relations only, so the intra-sentence
  scores are the like-for-like comparison. Inter-sentence prediction is additionally
  possible under the document-level formulation (one call covers all pairs) where per-pair
  prompting is cost-prohibitive — results pending.
- Costs per full evaluation (incl. resampling and CoT generation): CTB $7.49,
  ESC $13.19, MAVEN $24.15.

## Inference & aggregation (resampling analysis)

Vote-threshold sweep over the N=3 resampling passes of the three main runs
(predict causal iff ≥ t passes agree):

| Threshold | CTB | ESC | MAVEN-ERE |
|---|---|---|---|
| t=1 (union) | 36.1 / 55.0 / 43.6 | 31.8 / 83.5 / 46.1 | 31.8 / 75.8 / 44.8 |
| t=2 (majority, used) | 57.8 / 39.6 / **47.0** | 40.1 / 71.0 / **51.3** | 40.5 / 59.8 / **48.3** |
| t=3 (unanimous) | **76.7** / 30.9 / 44.0 | **46.9** / 54.5 / 50.5 | **49.0** / 41.7 / 45.1 |

- The majority vote with ties-to-norel is **empirically F1-optimal on all three datasets**
  (union costs 3.5–5.2 F1; unanimous costs 0.8–3.0).
- Mechanism: false positives are pass-specific while true positives are stable — on CTB,
  70% of union false positives appear in only one of three passes, whereas 72% of union
  true positives survive the majority. Per-pass hallucinations are ephemeral; the vote
  filters them.
- The threshold is a clean precision/recall knob: unanimous voting reaches P 76.7 on CTB
  (P 49.0 MAVEN) for precision-critical applications; union recall (55–84) bounds the
  recall ceiling of the demonstration ensemble.
- Each resampling pass uses a different demonstration set at temperature 0, so resampling
  acts as an **ensemble over demonstrations** rather than stochastic sampling.

## Error analysis: causal density

Documents ranked by causal density (gold intra causal pairs per event mention), equal-count
quartiles, pooled metrics per quartile:

| Quartile | docs (CTB/ESC/MAVEN) | gold pairs | CTB P/R/F1 | ESC P/R/F1 | MAVEN P/R/F1 |
|---|---|---|---|---|---|
| Q1 sparsest | 45/58/177 | 0 / 216 / 142 | 0.0 / 0.0 / 0.0 | 27.2 / 73.1 / 39.6 | 13.9 / 66.2 / 23.0 |
| Q2 | 46/58/178 | 15 / 330 / 517 | 31.2 / 33.3 / 32.3 | 38.6 / 76.1 / 51.2 | 30.6 / 63.2 / 41.3 |
| Q3 | 46/58/177 | 140 / 401 / 1000 | 61.6 / 37.9 / 46.9 | 46.8 / 70.8 / 56.3 | 43.0 / 62.5 / 51.0 |
| Q4 densest | 46/59/178 | 143 / 645 / 1728 | 74.1 / 42.0 / 53.6 | 46.7 / 67.4 / 55.2 | 55.5 / 56.3 / 55.9 |

- **Precision and F1 rise monotonically with causal density** on all three datasets
  (MAVEN P 13.9 → 55.5; CTB 0 → 74.1); recall declines only mildly. Contrary to the
  attention-dilution intuition, dense documents are the easy case; **sparse documents are
  the failure mode** — the model insists on finding causality where there is little or none
  (CTB's sparsest quartile, 45 documents with zero gold pairs, yields exclusively false
  positives).
- Gold mass is concentrated in dense documents (CTB: 283/298 pairs in Q3–Q4; MAVEN:
  2728/3387), so pooled micro metrics are dominated by the dense regime; a per-document
  macro view would be substantially lower.
- Practical implication: a sparse-document guard (or unanimous voting applied only to
  low-density documents) targets most remaining false positives.

## Backbone analyses

- **Cost-effectiveness** *(single runs per point)*: on the same MAVEN-ERE 50-doc setting,
  a mid-tier open model (deepseek-v4-pro) matches or beats the frontier model
  (gpt-5.5: F1 40.1 at $7.23) at F1 39.6–45.2 for **$0.66–1.12** — an order of magnitude
  better cost-per-F1. glm-5.2: 40.4 at $1.81; gpt-4o-mini: 24.3.
- **Synthetic-CoT demonstrations interact with backbone type** *(preliminary; mixed
  configs)*: reasoning-bearing demonstrations helped chat-tuned models (deepseek-v3.2 on
  CTB: +9.0 F1 over label-only demos, 13 vs 1 runs; glm-5.2 on MECI: +16.2, 1v1) but
  consistently *hurt* the reasoning-tuned qwen3-30b-thinking (MECI: −9.7, 13 vs 8 runs;
  ESC: −6.3), which also accounted for most output-format failures. In-context reasoning
  transcripts appear to interfere with models that have their own trained reasoning
  channel.
- **Demonstration selection** *(preliminary)*: random selection matched or exceeded
  TF-IDF and embedding retrieval at document scale (MAVEN, same setting: random 45.0–47.1
  vs embedding 37.5–46.6 vs TF-IDF 39.6), echoing at document level SERE's report that
  naive retrieval can underperform random at pair level. Larger demonstration pools
  (more per-document diversity) trended positive.

## Attribution note for the CTB result *(preliminary — older backbone)*

On CTB, a dataset-appropriate precision rule in the prompt ("causal label only when an
explicit causal marker connects the events" — CTB annotation is ~53% explicit-connective)
accounts for most of the gain: with the generic prompt the same scaffold scored F1 8.9–12.6
(below SERE), the marker rule alone reached ~40 zero-shot, and few-shot + synthetic CoT
added ~+3–9 on top (mean 44.0 over 10 repeats, deepseek-v3.2). The document-level format is
what makes such a rule enforceable over every pair at once; this decomposition should be
stated when presenting the CTB column.

## Reporting & significance facts

- Run-to-run variation on 50-document subsets is sd ≈ 2–4 F1 (10 identical-config repeats:
  range 39.8–51.0) even at temperature 0; single-run differences under ~4 F1 are not
  meaningful and multi-seed reporting is required for ablation deltas.
- Evaluation protocol: binary causal identification (direction excluded by problem
  definition); ordered-pair convention; failed documents scored as all-norel rather than
  excluded; CTB/ESC evaluated over their full corpora with k-fold separation between
  demonstration pool and evaluated documents; MAVEN-ERE evaluated on the 710-document test
  split with training-split demonstrations.
