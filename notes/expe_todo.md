# Experiment TODO (2026-07-19)

Priorities set with the author. Context that shapes this list:
- Direction of causal relations is **out of scope** (problem resolves to binary
  classification) — no direction-related experiments.
- SERE evaluates **intra-sentence only**, so the intra-only SCOTECI numbers are the fair
  comparison. Inter-sentence prediction is a *capability* of SCOTECI (pair-by-pair methods
  can't afford O(n²) calls), not a comparability gap.
- Pair-level controls are **not in the pipeline** — dropped.

Reference costs: full final-config runs were CTB $7.5 / ESC $13 / MAVEN $24; n=50 runs $1–3.5.
Run-to-run noise at n=50 is sd ≈ 2–4 F1 → every ablation arm needs ≥3 seeds.

## P0 — must address before submission (verify the claim or change the text)

### 1. Decontamination ablation
- **Claim at stake**: §Method / §Ablations — "decontamination ... which we show is
  load-bearing". Currently zero supporting runs.
- **Code change needed first**: decontamination-off is NOT reachable via config today —
  in `tools/few_shot.py::generate_cot` the decontaminate step runs unconditionally in both
  the 2-step and 3-step branches (`num_steps=1` silently behaves as 2-step). Add e.g.
  `few_shot.cot_generation.decontaminate: false` that returns the raw gold-guided response
  as the demo (and namespace the `_COT_CACHE` key with it).
- **Runs**: {decontam on, off} × {CTB, MAVEN} × 3 seeds, n=50, final v4-pro config
  (random k=3, num_steps as shipped). ~$15.
- **Decision rule**: |Δ| > ~4 F1 and consistent sign → keep the claim with numbers;
  otherwise rewrite as "a safeguard against answer-leakage phrasing (ablation: ±x F1)".
- This doubles as the decontam-off arm of the ablation grid's CoT axis (expe 3) — running
  it first loses nothing.

### 2. Explicit-marker ablation on CTB (v4-pro)
- **Claim at stake**: the CTB headline is attributed to the doc-level protocol + synthetic
  CoT; v3.2-era evidence (mined §4) says the `explicit_marker` prompt rule contributes ~+30 F1
  and the scaffold ~+3. Needs confirmation on the actual backbone, and disclosure either way.
- **Runs**: `dataset.prompt: causal_timebank_explicit_marker` vs
  `causal_timebank_no_marker` (created 2026-07-19: byte-identical to the marker prompt with
  only the marker-requirement parts removed — a clean single-variable ablation, unlike the
  June-era `causal_timebank_standard_eci_norule` which differs in more ways), v4-pro final
  config, n=50 × 3 seeds each (~$6); optionally one full-183 run of the winner's counterpart
  for the paper number (~$7.5).
- **Outcome**: feeds §4.2 (disclose the dataset-specific rule) + §Ablations; defensible
  either way (CTB is ~53% explicit-connective), but it must be measured and stated.

## P1 — strengthens the paper

### 3. Cross-dataset ablation grid (fills §6.1 wholesale)

One protocol for all ablations: **center point = the final recipe** (v4-pro, random k=3,
pool 100, 3-step CoT, rs3 distinct-fewshots, lean intra prompt, temp 0), **n=50 × 3 seeds ×
all three datasets (CTB/ESC/MAVEN)**, vary ONE axis at a time. Every cell is a `queue.yaml`
override; ~$1–3.5 per run at rs3 (≈⅓ at rs0 — decide once whether the grid runs at rs3 for
fidelity to the table or rs0 for cost, and say so in the paper).

The three seed repeats of the shared center point double as the variance estimate (absorbs
old P2-5), and every cell's final run gets the vote-threshold sweep for free
(`scripts/analysis/vote_threshold.py`).

Axes (first three are the priority):

| Axis | Arms (center in bold) | Config key(s) | Paper bullet it fills |
|---|---|---|---|
| k (number of few-shots) | 0 (=few-shot off), 1, **3**, 5 | `few_shot.enabled` / `few_shot.n_examples` | "k ∈ {1,3,5}" + "remove few-shot entirely" |
| Resampling N | 1, **3**, 5 (+9 on CTB only, cheap) | `experiment.resampling.n_runs` | §Inference resampling todo; N>3 also enriches the vote-threshold figure (any t ≤ N) |
| CoT demos | label-only (off), 2-step, **3-step**, decontam-off (P0-1 arm) | `few_shot.cot_generation.enabled` / `.num_steps` / new `.decontaminate` | "remove synthetic CoT" + "2- vs 3-step" + decontamination |
| Demo-set ensembling | distinct fewshots **on** / off at N=3; optionally off + temp 0.7 | `experiment.resampling.distinct_fewshots`, `model.temperature` | separates demo-ensemble gain from stochastic-resampling gain (mined §7) — currently conflated |
| Pool size (demo diversity) | 3, 20, **100** | `few_shot.pool_size` | pool-size/cost trade-off (paper only mentions cost) + cache-hit economics (mined §6) |
| Selection strategy | **random**, similarity (TF-IDF), bert embeddings | `few_shot.selection` | "random vs TF-IDF vs embedding" (mention-Jaccard: only if implemented) |
| Structured protocol | **steps prompt**, norule (direct JSON, demos kept) | `dataset.prompt` → `*_norule*` variants | "remove the structured protocol"; k=0 × steps covers "protocol alone" |
| Coherence rules block | **on** / off (prompt-text edit) | rules block in prompt YAMLs | "coherence rules on/off (in-prompt)" |

Full grid ≈ 19 non-center arms × 3 datasets × 3 seeds + 9 center runs ≈ **180 runs**
(~$200–600 at rs3, ~$70–200 at rs0). If budget-capped, run the priority axes (k, resampling,
CoT) across all datasets first (~60 runs) and the remaining axes on CTB + MAVEN only.
Reporting stays within the single-table constraint: prose deltas vs the center point, or one
ablation figure per axis.

### 4. Fixed-config model ladder (§6.3 scaling claim)
- Existing ladder points are confounded (pool/selection/steps vary). Freeze ONE config
  (final recipe: random k=3, 3-step CoT, rs3, lean intra prompt) and sweep backbones via
  `queue.yaml` overrides: gpt-5.5, deepseek-v4-pro, glm-5.2, deepseek-v3.2,
  qwen3-30b-thinking, gemini-flash-lite tier, gpt-4o-mini.
- MAVEN intra n=50 (+ optionally CTB n=50). ~$15–25 (gpt-5.5 dominates the bill).
- Deliverable: F1-vs-tier figure; also yields the cost-per-F1 point (mined §8) for free.
- Note: without a pair-level arm, frame §6.3 as "SCOTECI improves with backbone quality",
  not as a doc-vs-pair crossover.

### 5. Inter-sentence capability run (ESC and/or MAVEN)
- Reframe: not a comparability fix (SERE is intra-only) but a **SCOTECI-only capability** —
  document-level calls make inter-sentence prediction affordable where per-pair prompting
  is cost-prohibitive.
- **Runs**: final config with the non-intra prompts (`esl_standard_eci`,
  `maven_ere_standard_eci`), full eval sets (~$15–30 each). `binary_intra`/`binary_inter`
  metrics are already logged separately, so one run per dataset gives the whole analysis.
- Report intra (comparable to SERE) and inter (new capability) side by side; the current
  final runs have binary_inter = 0 *by prompt design*, so this is genuinely unmeasured.

## P2 — nice to have, not urgent

### 6. Variance estimation — absorbed by the grid
- Covered by the ablation grid's 3-seed center point (expe 3): those 9 runs ARE the
  significance/sd statement for the results section. Only run it standalone (~$10) if the
  grid is skipped. Near-duplicates already suggest sd ≈ 2–4 F1.

### 7. Teacher-vs-self CoT generation
- v4-pro-generated demos consumed by qwen3-30b-thinking (and/or gemini-flash-lite), MAVEN
  n=50 × 3 seeds (~$5). Tests the paper's open question (does a stronger generator lift a
  weaker reasoner) and probes mined §5 (CoT demos currently *hurt* the thinking-tuned model).

## Dropped / settled — no runs needed

- **Pair-level same-backbone control**: not in the pipeline (author decision). Adjust §6.2
  text accordingly (cost argument can rest on the O(n²)-calls arithmetic + measured SCOTECI
  token counts, and on §4's inter-sentence affordability point).
- **Union-voting confirmation**: settled from traces — majority is F1-optimal (mined §1).
- **ESC direction-mapping audit**: out of scope — direction is irrelevant under the binary
  problem definition.

## Analyses ready to drop into the paper (no new runs)

- **Vote-threshold sweep** (union / majority / unanimous, any t ≤ N):
  `scripts/analysis/vote_threshold.py <run>` — works on any MLflow run (trace-based,
  self-validating against the run's logged metrics). Result worth a paragraph + small
  figure in §Inference/§Analysis: majority is F1-optimal on all three datasets, unanimous
  is the precision mode (CTB P 76.7), union recall bounds the demo-ensemble recall ceiling
  (mined §1). Rerun it on the final runs of any new experiment for free.

## Text-only fixes (no runs, do alongside)

- Abstract/intro/prose still cite the June numbers (51.0/53.2 CTB; 35.5/36.6 MAVEN; 43.2 ESC)
  — update to the table's 48.1/61.1/39.6, 48.4, 51.7; "+31 F1" → "+28.1".
- §Implementation says TF-IDF selection + 2-step CoT; final runs use random + 3-step.
- Table caption: recast the `*` caveat using the SERE-intra-only fact (intra comparison is
  fair; inter-sentence results are additional capability, pending expe 4).
- §Inference: resampling = ensemble over demonstration sets at temp 0 (mined §7), ties→norel
  now empirically optimal (mined §1).
- MAVEN evaluated subset: state 710 test docs / 33k pairs; ESC: reconcile 233 vs 258 docs.
