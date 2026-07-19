# MLflow evidence audit for the SCOTECI paper (2026-07-19)

Source: `mlflow.db` (experiment `agentere`, 252 active runs, 2026-05-29 → 2026-07-19).
Metrics quoted as P/R/F1 ×100. "binary" = undirected causal-vs-norel collapse; "intra" = `binary_intra_*`.

---

## 1. Main results table (tab:main-results, SCOTECI row)

All three table numbers are traceable to specific runs. All use: deepseek/deepseek-v4-pro
(OpenRouter), temp 0, k=3 demos selected **random** from pool 100, synthetic CoT **3-step**,
resampling **N=3** with distinct few-shot sets per pass (tie → norel), tools disabled,
strict JSON with retries; failed docs scored as all-norel (not dropped).

| Dataset | Table value | Run | When | Scope | Matching metric |
|---|---|---|---|---|---|
| CTB 61.1/39.6/48.1 | ✅ exact | `204db64f` | 07-19 00:14 | all 183 docs, 756 pairs, 10-fold demo sep. | `binary_intra_*` |
| ESC 40.7/70.9/51.7 | ✅ exact | `be962b6e` | 07-17 19:18 | 233 docs (train split), 5-fold demo sep. | `binary_intra_*` |
| MAVEN 40.8/59.6/48.4 | ✅ exact | `6269c8a9` | 07-19 00:51 | 710 test docs, 33,144 pairs | `binary_intra_*` |

Notes / caveats on these runs:

- **The reported metric is undirected binary, intra-sentence subset.** For CTB this equals the
  dataset-level story (intra-only corpus; overall binary = 59.6/39.6/47.6, the small P gap is
  predictions the model emitted on cross-sentence pairs). For ESC and MAVEN the *full-document*
  binary scores of the same runs are much lower: **ESC 40.6/23.9/30.1**, **MAVEN 40.8/14.9/21.8** —
  but both runs used *intra-only prompts* (`*_intra_lean`), so these are not valid full-document
  results either. A genuine full-document (intra+inter) run does not exist for the final config.
  **Reframed 2026-07-19 (author input)**: SERE evaluates intra-sentence only, so the intra-only
  SCOTECI numbers are the *fair* comparison — the table caption's "not yet directly comparable"
  caveat should be recast, and inter-sentence prediction presented as a SCOTECI-only capability
  (see `expe_todo.md` P1-4).
- **MAVEN "needs an update as soon as the full prediction is given"**: the full intra run landed
  last night (`6269c8a9`, n=710). The table already contains its numbers.
- **CTB prompt is `causal_timebank_explicit_marker`**: "causes/causedby ONLY if an explicit
  causality marker directly connects the two events; no explicit marker → norel." This
  dataset-specific hard rule — not the generic precision-first block described in §4.2 — is what
  produces the precision jump on CTB. The paper's method section currently does not disclose it.
- CTB run has `dataset.rule_set: meci` (inert since tools are disabled, but worth cleaning).
- CTB: 2/183 docs failed all retries (scored all-norel). ESC/MAVEN: 0.
- The known reprompt/resample ordering bug (see memory) does **not** affect these runs:
  `reprompt.systematic=False` and `constrain_to_pair_list=False` (reshuffle inert).
- K-fold demo separation is correctly implemented (same-fold docs excluded from the demo pool),
  so evaluating ESC/CTB on the train split does not self-leak demonstrations.

### Stale numbers in abstract / intro / results prose (must be updated)

| Paper text | Source run | Status |
|---|---|---|
| "CTB F1 51.0, precision 53.2" (abstract, intro, §Results) | `723dac62`, 06-05, v3.2:nitro, **50-doc subset**, pool=3, **15/50 docs failed retries** | Superseded by 48.1/61.1/39.6. Also: that config was repeated ~10×, F1 range **39.8–51.0, mean 44.0, sd 4.0** — 51.0 was the best draw. |
| "+31 F1" | — | With the table numbers it is **+28.1** (48.1 vs 20.0). |
| "MAVEN best precision 35.5, F1 36.6" (§Results) | `fc027ab7`, 07-02, v3.2, n=50 | Superseded by 40.8 / 48.4. |
| "ESC 43.2" (§Results) | no exact match; closest 43.1 (`13f66cfe`, 07-15) | Superseded by 51.7 (which now *exceeds* SERE's 49.9, intra-caveat aside). |

Baseline rows (PaLM2/GPT-4o-mini/Gemini blocks) are reproduced from SERE/Dr.ECI papers — no
MLflow runs, consistent with the paper's statement that they were not re-run.

### Costs of the three table runs (for §Implementation / §Cost)

| Run | cost | tokens in / out (cache-read) |
|---|---|---|
| CTB full | $7.49 | 4.10M / 1.73M (0.74M) |
| ESC full | $13.19 | 7.44M / 5.51M (1.06M) |
| MAVEN full | $24.15 | 17.67M / 5.45M (3.73M) |

(Includes N=3 resampling and CoT-demo generation.)

---

## 2. Ablations (§6.1) — coverage

| Claimed ablation | MLflow evidence | Verdict |
|---|---|---|
| Remove synthetic CoT (label-only demos) | Only old-config runs: CTB v3.2 06-05: FS+noCoT **34.1** vs FS+CoT repeats mean **44.0** (1 seed vs 10); some MECI qwen pairs (noisy, both directions). Nothing on the final v4-pro config. | **Inadequate** |
| Remove few-shot entirely | CTB v3.2 06-05: no-FS **40.9** (single run; note: *above* the FS+noCoT 34.1). ESC/MAVEN v3.2-era no-FS runs exist but configs differ from final. | **Inadequate** |
| Remove structured protocol (`*norule*` prompts) | ESC `norule_intra` 07-15: intra 41.4 vs matched lean runs 40.8–44.4 (within noise, 1 seed). MAVEN `norule_coref` 07-02 (v3.2): 33.6 vs 34.8/36.6 matched. CTB norule only in pre-explicit-marker era (not comparable). | **Inadequate / inconclusive** |
| Selection: random vs TF-IDF vs Jaccard vs embedding | random / `similarity` (TF-IDF) / `bert` (all-roberta-large-v1) all present, but never in a controlled sweep — pool size and date confound every comparison. Mention-Jaccard **never implemented/run**. On MAVEN v4-pro n=50: random-pool3 45.0–45.2 vs bert-pool3 38.4 (random wins); ESC mixed. | **Partial, inadequate** |
| k ∈ {1,3,5} | k=3 everywhere; one k=6 MAVEN run (39.4 vs 38.4 matched k=3). k=1 and k=5 **never run**. | **Missing** |
| 2-step vs 3-step CoT | Two matched pairs: ESC 07-15 (bert, pool6): s2 **41.8** vs s3 **40.6**; MAVEN v3.2 (pool30): s2 **38.9** vs s3 **39.3**. Both within run-to-run noise (±2–3 F1). | **Inadequate** |
| Decontamination on/off | `num_steps=1` (gold-guided, no decontamination) was run once, on 1 document (gpt-5.5 smoke test). The paper's claim "we show \[decontamination\] is load-bearing" has **no supporting run**. | **Missing** |
| Coherence rules on/off | In all final runs `enable_tools=False` → the coherence checker tool was **inert**; rules exist only as prompt text. No matched on/off comparison exists (June-era tool runs use different prompts/models). | **Missing** |

Also: the paper's Implementation Details say **TF-IDF selection** and **2-step CoT** — the final
table runs actually used **random selection** and **3-step CoT** (config default is 2). Either fix
the text or re-run.

Run-to-run noise estimate (needed for any prose delta): identical CTB config repeated 10×
(06-05): sd ≈ 4 F1. ESC/MAVEN v4-pro n=50 near-duplicates differ by 2–4 F1. Most ablation deltas
above are smaller than this.

---

## 3. Document-level vs pair-level, same backbone + cost (§6.2)

- **True pair-level control (one pair per call): never run.** `prompts/esl_perpair.yaml` exists but
  no run used it. The closest thing is the `eci_extractor` *tool* pipeline (orchestrator delegates
  to per-mention sub-calls on a v3.2 extractor): CTB 06-25: F1 **39.0 / 38.2** vs doc-level
  same-era 43–51; ESC/MAVEN small-n runs mostly poor (5.5–24.5). This is per-mention agentic
  decomposition, not the paper's described control. The section calls this "the key internal
  comparison" — it is currently **missing**.
- Cost side: per-run `cost_usd` / token metrics are logged for everything (see §1 table), but with
  no pair-level counterpart there is nothing to plot against. **Partial.**

---

## 4. Scaling with model quality (§6.3)

Single-seed, partially-confounded ladder exists only on **MAVEN intra, n=50**:

gpt-5.5 40.1 · glm-5.2 40.4 · deepseek-v4-pro 38.4–45.2 (varies with pool/selection) ·
deepseek-v3.2 37.8–39.9 · qwen3-30b-thinking 35.9 · gemini-2.5-flash-lite 34.0 ·
gemini-3.1-flash-lite 30.1 · gpt-4o-mini 24.3.

Direction is plausible (bigger ⇒ better) but configs are not matched, each point is one seed, and
there is **no pair-level arm**, so the central claim — the doc-level advantage *widens* with model
quality (crossover) — has **no supporting evidence**. CTB anecdote: gpt-5.4-mini collapses to
9.5 F1 where v3.2 gets 35.8 (no-FS), consistent with "weak models over-generate", single seed.

---

## 5. Error analysis (§6.4)

Update 2026-07-19: the density analysis has been computed from the final runs' logs — see
`mlflow_mined_analyses.md` §3 (precision and F1 *rise*
with per-mention causal density; equal-count quartiles). A vote-threshold sweep re-pooled from
`traces.jsonl` is in §1 there (majority is F1-optimal; unanimous is a precision mode). Still
open: FP taxonomy (temporal-as-causal, reporting verbs, transitive flattening) and the
qualitative worked example. Note: `utils/reporting.py` was fixed on 2026-07-19 to also log
voted-but-rejected pairs; logs from *before* that fix cannot support aggregation-rule analyses
(re-pool from traces instead).

The raw material for the remaining items is fully logged for the three final runs:

- `<run>.json` → `results.per_pair_predictions`: every pair with gold, pred, vote counts,
  intra/inter flag (756 / 12.7k / 33.1k rows) → FP-taxonomy sampling is one script away.
- `results.per_doc_metrics` (183/233/710 docs) → precision-vs-gold-density scatter.
- Auto-logged PNGs: `density_per_pair_by_gold_density`, `sentence_distance`, `word_distance`
  histograms; per-doc causal-graph SVGs; `traces.jsonl` with raw reasoning (qualitative example).

**Status: available but not yet computed.**

---

## 6. Other claims / todos

- **Resampling statement (todo in §Results)**: final runs use N=3, majority vote, tie→norel,
  distinct few-shot sets per pass, constant prompt across passes. Matched rs0↔rs3 pairs are
  noisy: CTB +8.0 F1 (39.2→47.2), ESC −0.1, MAVEN −0.9 — one pair each; no basis yet for a
  significance/variance claim. A single N=9 run exists (MECI, 06-03).
- **Backbone claim (todo)**: final backbone = deepseek/deepseek-v4-pro via OpenRouter; CoT demos
  self-generated (generator = inference model). **Teacher-vs-self generation: never run.**
- **Same-backbone Base/CoT rows (todo)**: not run (the only gpt-4o-mini run is SCOTECI-style, not
  a Base/CoT reproduction).
- **MECI (todo: appendix?)**: 57 exploratory runs, extremely noisy — same v3.2 config on 06-29
  gave F1 37.8, 47.0 and 71.2; one run is tagged "relation inversion problem". Only full-test run:
  qwen3-30b, no-CoT, n=716, F1 37.8 (06-02). Best n=50: glm-5.2 FS+CoT 60.4; v3.2 FS+CoT 53.3.
  Per-language metrics (en/da/es/tr/ur) are logged. **Not publication-ready.**
- **Dataset facts**: ESC version confirmed 0.9 (`Nofing/EventStoryLine-0.9-standard-eci`), but the
  final run covers **233 docs**, not the 258 stated in §5.1 — reconcile. CTB = all 183 docs.
  MAVEN evaluated subset = **710 test docs / 33,144 pairs** (fills the "state sample size" todo).
  ESC and CTB are evaluated on the *train* split with k-fold demo separation; MAVEN on test with
  train demos — worth stating explicitly in the paper.
- **CoT generation cost (todo)**: not separately logged; total costs above include it. Would need
  either the `logs/cot_dumps` token counts or a re-run to separate.

## 7. Next experiments

Superseded 2026-07-19 by the author-triaged list in **`expe_todo.md`**. Summary:
**P0** = decontamination ablation (needs a small code change first — decontam-off is not
reachable via config; `num_steps=1` silently runs the 2-step protocol) and the CTB
explicit-marker ablation on v4-pro. **P1** = fixed-config model ladder; inter-sentence
capability run — reframed per the author: SERE evaluates intra-sentence only, so the intra
numbers are the fair comparison and inter-sentence prediction is a SCOTECI-only *capability*
(pair-level methods can't afford O(n²) calls), not a comparability gap. **P2** = variance
estimation; teacher-vs-self CoT. **Dropped**: pair-level control (not in the pipeline),
union-vote confirmation (settled from traces), ESC direction audit (direction is out of
scope — the problem definition is binary).

## 8. Bottom line

Solid, current evidence exists for the three SCOTECI table rows (binary, intra-sentence, with the
stated caveat) and for cost numbers. The abstract/intro/prose still carry June-era numbers from a
best-of-10, 50-doc run and must be rewritten around 48.1/61.1/39.6 (CTB), 51.7 (ESC intra),
48.4 (MAVEN intra). Every §6 analysis promised in the skeleton — ablation grid, decontamination,
coherence on/off, k sweep, pair-level control, model-ladder crossover, error taxonomy — is either
missing, single-seed, or confounded, and the undisclosed CTB explicit-marker prompt plus the
TF-IDF/2-step vs random/3-step description mismatches need fixing before submission.
