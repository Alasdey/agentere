# Analyses mined from MLflow that the paper misses (2026-07-19)

Counterpart to `mlflow_results_report.md`. Everything below is computed from runs already in
`mlflow.db` — from the three final table runs' `per_pair_predictions` logs and, where the log
is insufficient (§1), re-pooled from their raw `traces.jsonl`
(CTB `204db64f`, ESC `be962b6e`, MAVEN `6269c8a9`).

Status 2026-07-19: §1 and §3 verified/revised after review (§1 was corrected from a logging
artifact — see its correction note); §2 marked out of scope (direction is irrelevant under
the binary problem definition). Post-revision, the highest-impact items are §3 (density),
§4 (CTB attribution), §5 (CoT × backbone). Experiment follow-ups live in `expe_todo.md`.

---

## 1. Vote-threshold sweep: majority is empirically F1-optimal; unanimous is a precision knob

**Correction (2026-07-19):** an earlier version of this section claimed union voting beats the
majority vote by +6–13 F1. That was an artifact of a logging bias: `per_pair_predictions` only
contains gold ∪ final-majority pairs (`utils/reporting.py`, `reconstruct_pairwise_predictions`),
so pairs predicted causal in one pass, non-gold, and outvoted — exactly the FPs that union
voting adds — were invisible to the re-scoring. Do **not** run aggregation-rule analyses off
`per_pair_predictions` for runs logged before the fix; re-pool from `traces.jsonl` instead.

Reusable tool: `scripts/analysis/vote_threshold.py <run-uuid-prefix>` runs this sweep
(t = 1..N for any N resamples, all/intra/inter subsets) on any MLflow run — trace-based by
default, with a sanity check against the run's logged metrics; `--source log` works for runs
logged after the reporting.py fix and refuses earlier ones.

Trace-verified numbers (every inference pass of the three final runs parsed from
`traces.jsonl`, gold rebuilt via `load_hf_dataset_parsed`, binary intra; the majority row
reproduces the reported table within 0.1–1.1 F1, validating the method):

| Threshold | CTB P/R/F1 | ESC P/R/F1 | MAVEN P/R/F1 |
|---|---|---|---|
| ≥1 (union) | 36.1 / 55.0 / 43.6 | 31.8 / 83.5 / 46.1 | 31.8 / 75.8 / 44.8 |
| ≥2 (majority = shipped) | **57.8 / 39.6 / 47.0** | **40.1 / 71.0 / 51.3** | **40.5 / 59.8 / 48.3** |
| =3 (unanimous) | 76.7 / 30.9 / 44.0 | 46.9 / 54.5 / 50.5 | 49.0 / 41.7 / 45.1 |

Three usable results:

- **The majority-with-ties-to-norel design is empirically F1-optimal on all three datasets** —
  the paper currently asserts it is "precision-preserving"; this sweep turns that into a
  measured result (union costs 3.5–5.2 F1, unanimous costs 0.8–3.0).
- **False positives are pass-specific, true positives are stable**: on CTB, 70% of union FPs
  appear in only 1 of 3 passes while 72% of union TPs survive the majority. Per-pass causal
  hallucinations are ephemeral and the vote filters them — a clean mechanistic story for §Inference.
- **Unanimous voting is a real precision mode** (CTB P 76.7, MAVEN 49.0, ESC 46.9) and union
  recall (55.0 / 83.5 / 75.8) measures the recall ceiling across demonstration sets — useful
  bounds for the analysis section, one config flag away from a headline variant.

## 2. Direction consistency — OUT OF SCOPE (kept for the record only)

Author decision (2026-07-19): the direction of a causal relation is not part of the problem
definition — everything resolves to binary classification — so none of this belongs in the
paper and no fix is needed. For the record, since directed metrics exist in MLflow: among
binary-TP pairs of the final runs, predicted direction disagrees with the stored directed
label 55.1% of the time on ESC vs 0.0% (CTB) / 1.9% (MAVEN); this (plus 7 old runs with
`micro_f1=0`, `binary_f1` up to 40.9, one tagged "relation inversion problem") is why
directed `micro_*` metrics look low on ESC and why binary metrics are the ones to read.

## 3. Precision *rises* with document density — the paper's hypothesis is inverted

§Error-analysis plans to test "does precision degrade with the number of gold pairs per
document?" The per-pair logs answer: **no — the opposite.**

Method (revised 2026-07-19): documents ranked by causal density = gold intra causal pair rows
per event mention (mention counts from the dataset's `mentions_map`), split into equal-count
quartiles (~46/58/178 docs per bin for CTB/ESC/MAVEN), pooled binary-intra P/R per bin.
(An earlier version used fixed gold-count bins with very unequal document counts and no
per-mention normalization; the quartile version shows the same trend, now monotone.)

Gold = unordered gold intra causal pairs in the bin (the harness counts each in both
directions; row counts are 2×).

| Quartile | docs (CTB/ESC/MAVEN) | gold (CTB/ESC/MAVEN) | CTB P/R/F1 | ESC P/R/F1 | MAVEN P/R/F1 |
|---|---|---|---|---|---|
| Q1 (sparsest) | 45 / 58 / 177 | 0 / 216 / 142 | 0.0 / 0.0 / 0.0 (only FPs) | 27.2 / 73.1 / 39.6 | 13.9 / 66.2 / 23.0 |
| Q2 | 46 / 58 / 178 | 15 / 330 / 517 | 31.2 / 33.3 / 32.3 | 38.6 / 76.1 / 51.2 | 30.6 / 63.2 / 41.3 |
| Q3 | 46 / 58 / 177 | 140 / 401 / 1000 | 61.6 / 37.9 / 46.9 | 46.8 / 70.8 / **56.3** | 43.0 / 62.5 / 51.0 |
| Q4 (densest) | 46 / 59 / 178 | 143 / 645 / 1728 | **74.1** / 42.0 / **53.6** | 46.7 / 67.4 / 55.2 | **55.5** / 56.3 / **55.9** |

Precision climbs monotonically with density on all three datasets (MAVEN 13.9 → 55.5 across
quartiles of 177+ docs each; CTB 0 → 74.1), and **F1 climbs with it** (CTB 0 → 53.6,
MAVEN 23.0 → 55.9, ESC plateaus 56.3 → 55.2): the precision gain dominates the mild recall
decline. Note CTB's gold mass sits almost entirely in Q3–Q4 (283 of 298 pairs), so its Q1–Q2
FPs barely dent the headline metric despite the 0-precision sparsest quartile. Sparse documents
are the *precision* problem — the model insists on finding something where there is little or
nothing (CTB's sparsest quartile yields exclusively false positives) — and dense documents are
at worst a mild *recall* problem. This reframes the "attention dilution" limitation and
suggests a cheap fix: a sparse-doc guard (allow-empty-output emphasis, or the unanimous-vote
rule from §1 applied only to low-density documents).

## 4. On CTB, the win comes from the explicit-marker rule, not the scaffold

Decomposition available from June runs (v3.2, n=50, same eval protocol):

- standard/norule prompt, any scaffold combo: F1 **8.9–12.6** (P 5–7) — *below* SERE's 20.0;
  the generic doc-level protocol does not fix hallucination on CTB.
- `explicit_marker` prompt, zero-shot no-CoT: F1 **40.9** (single run; 35.8 second run)
- `explicit_marker` + FS + synthetic CoT: mean **44.0** over 10 repeats (sd 4.0)

So: prompt rule ≈ +30 F1, few-shot+CoT ≈ +3 on top, and label-only demos actually *hurt*
(34.1 < 40.9 no-few-shot). The paper currently attributes the CTB result to the doc-level
protocol + synthetic CoT; the honest story is "a dataset-specific explicit-marker gate does the
heavy lifting on CTB" — which is defensible (CTB's annotation is ~53% explicit-connective) but
must be disclosed, and the ablation section should measure exactly this decomposition on v4-pro.

## 5. Synthetic-CoT demos help chat models and *hurt* the reasoning-tuned model

Matched FS runs, CoT-demos on vs off (micro F1 means, same dataset/model/prompt):

| Backbone | Dataset | CoT demos | label-only | Δ |
|---|---|---|---|---|
| deepseek-v3.2 | CTB explicit-marker | 43.1 (n=13) | 34.1 (n=1) | **+9.0** |
| glm-5.2 | MECI | 60.4 (n=1) | 44.2 (n=1) | **+16.2** |
| qwen3-30b-**thinking** | MECI | 25.6 (n=13) | 35.3 (n=8) | **−9.7** |
| qwen3-30b-**thinking** | ESC | 6.0 (n=1) | 12.3 (n=3) | −6.3 |
| gpt-5.4-mini | MECI | 21.2 (n=1) | 22.1 (n=1) | ≈0 |

The thinking-tuned qwen also produced 5 of the 7 fully format-collapsed runs. Plausible story:
models with their own RL-trained reasoning channel are destabilized by in-context reasoning
transcripts, while chat models benefit. This is a genuinely novel, publishable observation
(and directly motivates the still-unrun teacher-vs-self generation experiment).

## 6. Random demo selection ≥ retrieval — SERE's own finding, already replicated

The paper lists "random can beat naive retrieval" as a hypothesis *to test*; the data already
supports it at document scale (MAVEN v4-pro n=50 intra, binary-intra F1):

- random: 45.0–47.1 (4 runs, incl. pool 3) · bert embeddings: 37.5–46.6 (8 runs) ·
  TF-IDF similarity: 39.6 (1 run)

And the final table runs themselves use **random** selection (while §Implementation says
TF-IDF). Related trend the paper never discusses: **demo-pool diversity** helps —
bert pool 3 → pool 9: mean ≈ 41 → 44; random pool 20/100 best overall. Trade-off worth one
sentence: diverse per-doc demos lower prompt-cache reuse (final runs read only 18–21% of
input from cache); fixed demo sets would be materially cheaper but slightly worse.

## 7. Resampling is actually a demonstration ensemble — and the paper describes it wrong

At temperature 0 with `constrain_to_pair_list=False`, the prompt is *identical* across the 3
passes; pair-order shuffling (the stated purpose of resampling in §Inference) never fires.
What differs per pass is the few-shot set (`distinct_fewshots=True`). So "resampling with
majority vote" is really *ensembling over demonstration sets* plus provider nondeterminism.
This reframing explains why passes disagree at all at temperature 0, and per §1 the
disagreement is asymmetric — FPs are pass-specific, TPs stable — which is why the vote
filters hallucinations. Fixes the §Inference text (pair-order shuffling is not the mechanism).
Observed effect of rs3 vs rs0 in matched pairs: CTB +8.0, MAVEN +0.9…+4.7, ESC ≈ 0.

## 8. Cost-effectiveness: the mid-tier model matches the frontier model at ~1/10 cost

Same MAVEN n=50 intra setting: gpt-5.5 F1 40.1 at **$7.23** vs deepseek-v4-pro 39.6–45.2 at
**$0.66–1.12**; glm-5.2 40.4 at $1.81; gemini-flash-lite tier 30–34 at ~$0.1–0.3;
gpt-4o-mini 24.3. For the cost section this is a stronger, already-measured claim than the
missing pair-level comparison: SCOTECI on a mid-tier open model beats the frontier model on
cost-per-F1 by an order of magnitude.

## 9. Robustness by backbone (supports the strict-JSON protocol discussion)

Docs failing all parse retries, per run: v3.2:nitro **3.3/run** (291 over 87) vs v4-pro
**0.1/run** (4 over 38) vs qwen-thinking 0.6/run. The v3.2→v4-pro upgrade nearly eliminated
parse failures — worth one line where the paper discusses JSON parsing/retries, and it
confounds v3.2-era ablations (failed docs score all-norel).

## 10. Minor threads

- **MECI per-language**: consistent ordering Urdu ≥ English > Turkish ≈ Danish > Spanish
  (e.g. glm-5.2: ur 74.1, en 73.0, tr 58.9, da 56.3, es 51.9; v3.2: ur 72.1, en 59.8, es 40.0).
  Urdu beating English is a nice counter-intuitive nugget if MECI goes in the appendix.
- **Temp-0 ≠ deterministic**: identical configs differ by up to 11 F1 (CTB repeats sd 4.0) —
  any prose delta under ~4 F1 needs multiple seeds; also an argument for reporting the
  resampling ensemble as the primary configuration.
- **`esl_standard_eci_rewrite` prompt** (document-rewrite pass before reasoning, June 16):
  best ESC binary F1 of its era (27.4 vs 22.1 standard), then abandoned — cheap to revisit.
