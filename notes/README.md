# Reading guide — SCOTECI experiment notes

Entry point for someone discovering this project. It gives the minimum context needed to
read the three analysis documents in this folder, written 2026-07-17 → 2026-07-19.

## What this project is

**SCOTECI** (working name, paper skeleton in `paper_v2/main.tex`) is an LLM prompting method
for **Event Causality Identification (ECI)**: given a document with marked event mentions
(`<ei3 acquired>`), decide for event pairs whether a causal relation holds. Two ideas:

1. **Document-level prompting** — ONE LLM call labels every pair of a document jointly
   (prior LLM work — Dr.ECI, SERE — prompts once *per pair*), under a structured in-prompt
   reasoning protocol (mention pre-filter → per-pair arguments → ranking → final JSON).
2. **Synthetic chain-of-thought few-shots** — demonstrations are whole neighbour training
   documents, each equipped with LLM-*generated* reasoning: the generator sees the gold
   labels ("gold-guided"), then the reasoning is rewritten to hide that ("decontamination").
   2-step = gold-guided → decontaminate; 3-step = blind attempt → gold correction →
   decontaminate.

No fine-tuning. Baselines (Base/CoT/Dr.ECI/SERE on GPT-4o-mini and Gemini-1.5-pro) are
reproduced from the SERE paper, not re-run.

## Datasets, metrics, and conventions (needed to read any table here)

- **Datasets**: Causal-TimeBank (**CTB**, 183 news docs, sparsest positives), EventStoryLine
  0.9 (**ESC**, 233 docs evaluated), **MAVEN-ERE** (Wikipedia, 710 test docs evaluated).
  MECI (multilingual) exists in the runs but is exploratory only.
- **Binary metrics are the ones that matter.** Labels are directed (`causes`/`causedby`/
  `norel`) in the pipeline, but the problem definition is binary (causal vs not) — direction
  is out of scope by author decision. Read `binary_*` metrics; directed `micro_*` can look
  much lower (especially ESC) for irrelevant reasons.
- **intra / inter**: pairs within one sentence vs across sentences. SERE (the main baseline)
  evaluates **intra-sentence only**, so intra numbers are the fair comparison;
  inter-sentence prediction is a capability only the document-level method can afford.
  The current headline numbers are `binary_intra_*`.
- Pairs are counted **ordered** (each unordered pair appears in both directions), so "gold
  rows" are 2× the number of unordered gold pairs.
- All scores in these notes are ×100 (P/R/F1).

## The experiment infrastructure

- `main.py` + `config.yaml` run an evaluation; `queue.yaml` + `run_queue.py` batch runs as
  config overrides. Prompts live in `prompts/*.yaml` (per dataset, many variants).
- Every run logs to **MLflow** (`mlflow.db` SQLite + `mlruns/` artifacts): all config as
  params, metrics (binary/micro × all/intra/inter), and artifacts — a full report JSON
  (per-pair predictions with resampling vote counts, per-doc metrics), `traces.jsonl`
  (every raw LLM conversation), causal-graph SVGs, density histograms.
- **Resampling**: the final runs do N=3 passes per document and take a per-pair majority
  vote (ties → norel). At temperature 0 with distinct few-shot sets per pass, this is
  effectively an *ensemble over demonstration sets*, not stochastic sampling.
- **The three "final runs"** (the paper's Table-1 SCOTECI row; deepseek-v4-pro, random k=3
  demos from pool 100, 3-step CoT, N=3 resampling):
  | Dataset | run id | date | binary-intra P/R/F1 |
  |---|---|---|---|
  | CTB | `204db64f` | 07-19 | 61.1 / 39.6 / 48.1 |
  | ESC | `be962b6e` | 07-17 | 40.7 / 70.9 / 51.7 |
  | MAVEN | `6269c8a9` | 07-19 | 40.8 / 59.6 / 48.4 |

## The documents, in reading order

0. **`paper_ready.md`** — the distillation: only the validated, citable results and
   analyses, organized by the paper section they feed, stripped of implementation and
   history. If you just need "what can go in the paper", read only this one.
1. **`mlflow_results_report.md`** — the audit: every claim in the paper draft mapped to the
   MLflow run(s) that do (or don't) support it. Read this first — it establishes which
   numbers are current (the table) vs stale (abstract/prose still cite June-era runs), which
   ablations exist, and what's missing. Its §7 now just points to `expe_todo.md`.
2. **`mlflow_mined_analyses.md`** — the reverse pass: analyses the *data* supports that the
   paper doesn't use. Ranked findings; the load-bearing ones are §3 (precision *rises* with
   per-mention causal density — the paper's hypothesis was backwards), §4 (on CTB an
   explicit-causal-marker prompt rule, not the scaffold, drives the headline), §5 (synthetic
   CoT demos help chat models, *hurt* the reasoning-tuned qwen), §1 (vote-threshold sweep:
   the shipped majority vote is F1-optimal; unanimous is a precision knob).
3. **`expe_todo.md`** — the author-triaged experiment plan derived from 1+2: P0
   (decontamination ablation — needs a small code change first; CTB explicit-marker
   ablation), P1 (cross-dataset ablation grid over k / resampling / CoT / ensembling / pool /
   selection / protocol / coherence; fixed-config model ladder; inter-sentence capability
   run), P2, dropped items, and text-only paper fixes.

Older notes (`ablations.md`, `routing_methods_eci_claude_pilot_and_scaleup_plan.md`) predate
this analysis cycle and are not part of it.

## Corrections trail (why some sections say "revised")

These documents were actively reviewed and two findings were corrected — kept in place with
correction notes rather than silently rewritten:

- **Union-voting retraction** (`mlflow_mined_analyses.md` §1): an initial log-based analysis
  claimed union voting (≥1 of 3 passes) beats the majority vote by +6–13 F1. Root cause: the
  per-pair prediction log only contained gold ∪ final-prediction pairs, hiding exactly the
  false positives union adds. Re-pooling from raw `traces.jsonl` showed union is *worse*.
  `utils/reporting.py` was fixed on 2026-07-19 to also log voted-but-rejected pairs —
  **logs from before that date cannot support vote/aggregation analyses** (re-pool from
  traces instead).
- **Density analysis revision** (§3): re-done with per-mention density and equal-count
  document quartiles instead of ad-hoc fixed bins; the conclusion survived and strengthened.
- **Direction finding demoted** (§2): direction confusion stats were computed, then marked
  out of scope — the problem definition is binary.

## Tools

- `scripts/analysis/vote_threshold.py <run-uuid-prefix>` — vote-threshold sweep
  (union/majority/unanimous, any t ≤ N) for any MLflow run. Trace-based by default,
  self-validating (prints its reconstruction next to the run's logged metrics and warns on
  deviation); `--source log` only accepts post-fix runs.
- Other one-off analyses in `scripts/analysis/` (summary export, error analyses, histograms).

## Gotchas for newcomers

- **Temp 0 is not deterministic**: identical configs differ by up to 11 F1 on 50-doc subsets
  (sd ≈ 2–4 F1). Never trust a single-run delta under ~4 F1.
- ESC and CTB are evaluated on the *train* split (all there is), with k-fold separation so a
  document's demos never come from its own fold. MAVEN is evaluated on test with train demos.
- ESC/MAVEN final runs use **intra-only prompts**; their `binary_inter` is 0 by design, not
  ability. n=50 runs are quick-iteration subsets; only n=183/233/710 runs are "full".
- The paper draft's abstract/intro/prose still carry June-era numbers (CTB 51.0/53.2 from a
  best-of-10, 50-doc run); the table is the current source of truth.
- Params like `encoder.filter_norel.*` and `experiment.tools` are logged for every run but
  **inert** unless `experiment.enable_tools` is true (it is false in all final runs).
