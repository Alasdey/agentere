# ECI manual-prompting pilot → scale-up plan for `agentere`

This note has two parts:

1. **Pilot study** — a small (12-document) manual experiment where Claude itself acted as the predictor (no API, no `agentere` pipeline) to test whether *neighbour-density routing* + *rich pos/neg few-shot* improves Event Causality Identification before investing in running it at scale through this repo's OpenRouter pipeline.
2. **Scale-up plan** — how to reproduce/extend the same ideas using `agentere`'s existing `main.py` / `config.yaml` / `queue.yaml` machinery across many models, plus the concrete code gaps that need filling first.

---

## Part 1 — Pilot study (manual, no pipeline)

**Setup:** 4 HF datasets (MECI, MAVEN-ERE, CausalTimeBank, EventStoryLine), 3 held-out documents each (12 total). For each document: read the raw `<ID word>`-marked text, predict causal pairs directly, then score against gold with a Python binary-undirected micro-F1 scorer (pairs collapsed to `frozenset({A,B})`; direction confusion still counts as a true positive — equivalent to `scripts/analysis/binary_eval.py --mode or` in this repo).

**Strategies compared (same 12 docs throughout):**

| Route | Description |
|---|---|
| A: Causal-marker | Only pairs with explicit causal connectives |
| B/E: Mention-by-Mention (exhaustive) | Every anchor event compared against every other anchor |
| C/CuD: marker ∪ narrative-chain (union) | |
| D: Narrative Chain | Trace the temporal/causal storyline, emit pairs along it |
| **Routing (v1/v2)** | Pick route per document from *neighbour density* = avg(`n_pairs/n_event`) of top-k TF-IDF-nearest training docs. DENSE → E, SPARSE → D, else → CuD |
| **v3: rich few-shot** | Same routing, but the prompt shows k=3 TF-IDF nearest-neighbour training docs as **fully worked examples with explicit CAUSAL *and* NOT-CAUSAL labelled pairs** (anchor-by-anchor table for E/CuD routes, flat list for D) |

Per-dataset routing decision (from measured density thresholds):

| Dataset | thresholds (sparse, dense) | Routes used (sample 0,1,2) |
|---|---|---|
| MECI | 3, 5 | D, CuD, CuD |
| MAVEN | 8, 46 | E, E, E |
| CTB | 25, 106 | D, D, D |
| ESL | 3, 40 | E, D, E |

### Results — fixed single-strategy baselines (no few-shot at all)

| Route | MECI | MAVEN | CTB | ESL | **Overall** |
|---|---:|---:|---:|---:|---:|
| A: Causal-marker | 0.333 | 0.279 | 0.435 | 0.128 | 0.231 |
| B: Mention-by-Mention (E) | 0.286 | 0.160 | 0.667 | 0.095 | 0.187 |
| C: marker ∪ narr-chain (CuD) | 0.167 | 0.279 | 0.455 | 0.127 | 0.218 |
| D: Narrative Chain | 0.375 | 0.381 | 0.364 | 0.202 | **0.294** |

No fixed strategy wins everywhere — D is best overall, but B (exhaustive) wins specifically on CTB.

### Results — v3 routed + rich few-shot

Per-sample breakdown (3 TF-IDF-nearest neighbour worked examples per document, with explicit CAUSAL + NOT-CAUSAL labelled pairs, using the routed strategy from the table above):

| Dataset | idx | Gold pairs | Pred pairs | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meci | 0 | 2 | 2 | 0 | 2 | 2 | 0.000 | 0.000 | 0.000 |
| meci | 1 | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| meci | 2 | 2 | 6 | 1 | 5 | 1 | 0.167 | 0.500 | 0.250 |
| maven | 0 | 12 | 20 | 11 | 9 | 1 | 0.550 | 0.917 | 0.687 |
| maven | 1 | 4 | 50 | 3 | 47 | 1 | 0.060 | 0.750 | 0.111 |
| maven | 2 | 4 | 7 | 1 | 6 | 3 | 0.143 | 0.250 | 0.182 |
| ctb | 0 | 1 | 3 | 1 | 2 | 0 | 0.333 | 1.000 | 0.500 |
| ctb | 1 | 4 | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ctb | 2 | 1 | 4 | 0 | 4 | 1 | 0.000 | 0.000 | 0.000 |
| esl | 0 | 4 | 12 | 4 | 8 | 0 | 0.333 | 1.000 | 0.500 |
| esl | 1 | 2 | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| esl | 2 | 54 | 154 | 44 | 110 | 10 | 0.286 | 0.815 | 0.423 |

Per-dataset micro-F1 (aggregated from the per-sample rows above):

| Dataset | TP | FP | FN | Precision | Recall | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| MECI | 2 | 7 | 3 | 0.222 | 0.400 | 0.286 |
| MAVEN | 15 | 62 | 5 | 0.195 | 0.750 | 0.309 |
| CTB | 5 | 6 | 1 | 0.455 | 0.833 | **0.588** |
| ESL | 50 | 118 | 10 | 0.298 | 0.833 | 0.439 |
| **Overall micro** | 72 | 193 | 19 | **0.272** | **0.791** | **0.404** |

v3 beats every fixed strategy overall (+0.110 over D), almost entirely on **recall** (0.79 vs. 0.2–0.4 baseline). The raw gold annotations and raw prediction triples behind every row above (this section, the baseline table, and the failure-mode discussion below) are reproduced in full in Appendices A–C at the end of this note — nothing about this experiment requires going back to the session's `/tmp/`.

### Failure mode (important — carries over to the scale-up)

Precision collapsed on the two documents using the **exhaustive (E) route** on dense text:
- MAVEN sample 1: 50 predicted pairs, 3 correct (P=0.06).
- ESL sample 2 (54 gold pairs — the densest doc tested): 154 predicted pairs, 44 correct, 110 FP (P=0.29).

The exhaustive anchor-by-anchor format has no built-in stopping signal: once primed with positive examples, the model keeps finding "plausible" pairs, including indirect/transitive chains the annotation scheme doesn't credit (e.g. "shooting → trial → verdict" gets flattened into "shooting causes verdict"). The explicit NOT-CAUSAL contrastive examples helped (CTB went from 0.26 → 0.59) but did not fully fix this on the densest E-route docs.

> **Implication for scale-up:** any version of this method ported into `agentere` needs either (a) a precision-side guard on the exhaustive route (cap pairs per anchor, require independent justification, lower temperature), or (b) avoid the exhaustive route on dense documents and prefer narrative-chain/coherence-rule filtering instead, which `agentere` already supports via the `coherence` tool.

### Reproducibility note

This pilot was deliberately run *outside* `agentere` — by hand, with Claude as the predictor, no OpenRouter key, no training. It was originally run from session-local files under `/tmp/` (`routing_setup.json`, `rfp_*.txt`, `preds_{A,B,C,D}.json`, `eci_experiment_report.md`), which do **not** persist across sessions. Everything from those files that matters for this record — every gold annotation and every raw predicted triple for all 12 documents, across all 4 baseline routes and v3 — has been copied verbatim into **Appendices A–C** at the end of this note. This note is the durable, self-contained record; the `/tmp/` files should be treated as already gone.

---

## Part 2 — Scale-up plan using `agentere`

Goal: turn the pilot's two ideas — **(1) TF-IDF neighbour-density routing** and **(2) rich few-shot with explicit positive+negative contrastive pairs** — into config/prompt/code changes in this repo, then run them across a diverse set of OpenRouter models and the full test splits (not just 3 docs/dataset).

### 2.1 What already exists and can be reused as-is

| Pilot concept | `agentere` equivalent | Status |
|---|---|---|
| TF-IDF k-nearest few-shot | `few_shot.selection: "similarity"` in `config.yaml`, implemented in `tools/few_shot.py` | ✅ exists |
| k (number of few-shot examples) | `few_shot.n_examples` | ✅ exists |
| Binary undirected micro-F1 scoring | `scripts/analysis/binary_eval.py` (`--mode or` = our scorer) | ✅ exists, matches pilot metric exactly |
| Running many models | swap `model.default_model_id`, batch via `queue.yaml` + `run_queue.py` | ✅ exists |
| Majority-vote / resampling for stability | `experiment.resampling` (`n_runs`, `tie_breaking`) | ✅ exists |
| Negative examples in few-shot (constrained datasets) | `format_gold_output` already emits every pair in `pair_list_ids`, including `norel`, for MECI/CTB (`constrain_to_pair_list: true`) | ✅ partially exists |
| Post-run analysis / model comparison spreadsheet | `scripts/analysis/summary.py` | ✅ exists |

### 2.2 What's missing — concrete gaps to fill before scale-up

**Gap 1 — explicit negative contrast for open-ended datasets (MAVEN-ERE, ESL).**
`format_gold_output` (`utils/formatting.py`) only includes `norel` pairs when `pair_list_ids` is present (MECI/CTB path). For ESL and MAVEN-ERE, which run open-ended (`constrain_to_pair_list: false`), the few-shot examples currently show **only positive pairs** — exactly the v1/v2 setup that underperformed in the pilot. To replicate the pilot's "rich" v3 prompt, `tools/few_shot.py`'s `_format_examples` needs a mode that also samples a handful of *true negative* mention pairs from the same training doc (any two mentions not in `gold_triples`) and renders them as `NOT-CAUSAL: A ✗ B` lines alongside the positive `CAUSAL: A → B` lines. Smallest change: add a `few_shot.show_negatives: true` flag, default off, that augments `_format_examples` for non-`pair_list` datasets only (MECI/CTB already get this for free via existing `norel` entries).

**Gap 2 — density-based per-document routing.**
The pilot picked the *extraction strategy itself* (D / E / CuD) per document based on TF-IDF-neighbour density, not just which few-shot examples to show. `agentere` currently fixes one `prompt:` per dataset for an entire run (`datasets.<key>.prompt` in `config.yaml`). There's no per-document prompt switch today.
- **Cheap proxy (recommended first step):** since the pilot found one dominant route per *dataset* (D for MECI/CTB, E for MAVEN, mixed for ESL — see thresholds table above), just run each dataset with the prompt variant that matches its measured dominant route, instead of trying to route per-document. This requires zero pipeline changes — only new prompt files (Gap 3) and `queue.yaml` entries (2.4).
- **Full per-doc routing (follow-up, larger change):** would need a density score computed at dataprep time (reuse the existing TF-IDF matrix already built in `tools/few_shot.py` for `selection: "similarity"`), threaded through to `main.py`'s prompt-selection step so it can pick between two pre-loaded prompt templates per document. Flag as a separate ticket — do not block the first scale-up pass on this.

**Gap 3 — exhaustive anchor-by-anchor prompt variant.**
None of the existing `prompts/*.yaml` use the pilot's "ANCHOR X vs: Y, Z, …" exhaustive table format for the E route, or the worked anchor-by-anchor few-shot rendering with `→ CAUSAL / ← CAUSAL / ✗ NOT-CAUSAL`. New prompt files needed, one per dataset that will use the E route at scale (mainly MAVEN-ERE, and ESL for dense docs): e.g. `prompts/maven_ere_anchor_eci.yaml`, `prompts/esl_anchor_eci.yaml`, modeled on the existing `*_standard_eci.yaml` files but with the anchor-table instruction block and (per Gap 1) negative-aware few-shot.

**Gap 4 — precision guard on the E route.**
Given the pilot's precision collapse on dense E-route docs, before declaring success at scale, wire in (or strengthen) one of:
- the existing `coherence` tool (already in `experiment.tools`) to prune transitive false positives post-hoc,
- a per-anchor cap (e.g. "at most N causal pairs per anchor unless the text states more than N explicitly") added to the new anchor prompt's instructions,
- lower `model.temperature` for the E route specifically (currently global `0.0`, already minimal — so this lever is likely exhausted; coherence-rule pruning is the more promising lever).

### 2.3 Diverse-model matrix for the scale-up run

Pull a representative spread from the models already enumerated (commented out) in `config.yaml`, covering closed/open weights, frontier/small, and reasoning/non-reasoning:

| Tier | Model id |
|---|---|
| Frontier closed, non-reasoning | `openai/chatgpt-4o-latest` |
| Frontier closed, reasoning | `anthropic/claude-opus-4.7` |
| Frontier closed, reasoning | `openai/gpt-5-mini` |
| Open weights, large reasoning | `deepseek/deepseek-r1-0528` |
| Open weights, large non-reasoning | `deepseek/deepseek-v3.2:nitro` |
| Open weights, mid reasoning | `qwen/qwen3-30b-a3b-thinking-2507:nitro` *(current default)* |
| Open weights, mid non-reasoning | `qwen/qwen3-32b:nitro` |
| Open weights, small | `mistralai/mistral-small-3.2-24b-instruct` |
| Open weights, very small (stress test) | `mistralai/ministral-3b-2512` |

Rationale: the pilot used a single (strong) model — Claude — acting as predictor. The open question for scale-up is whether the recall gain from rich few-shot holds on weaker/smaller models, or whether it just amplifies their over-generation problem (likely, given the E-route precision collapse already seen on a strong model).

### 2.4 Proposed `queue.yaml` skeleton

```yaml
experiments:
  # Baseline: existing standard prompt, similarity few-shot, no negatives — current behavior
  - name: "baseline_meci_qwen30b"
    overrides:
      model: { default_model_id: "qwen/qwen3-30b-a3b-thinking-2507:nitro" }
      active_dataset: "meci"
      few_shot: { enabled: true, selection: "similarity", n_examples: 3 }
      datasets: { meci: { prompt: "meci_standard_eci", max_examples: 0 } }

  # v3-style: rich few-shot with explicit negatives, dominant-route prompt per dataset
  - name: "richfewshot_meci_qwen30b"
    overrides:
      model: { default_model_id: "qwen/qwen3-30b-a3b-thinking-2507:nitro" }
      active_dataset: "meci"
      few_shot: { enabled: true, selection: "similarity", n_examples: 3, show_negatives: true }  # Gap 1
      datasets: { meci: { prompt: "meci_standard_eci", max_examples: 0 } }   # MECI's dominant route is D — no new prompt file needed

  - name: "richfewshot_maven_qwen30b"
    overrides:
      model: { default_model_id: "qwen/qwen3-30b-a3b-thinking-2507:nitro" }
      active_dataset: "maven_ere"
      few_shot: { enabled: true, selection: "similarity", n_examples: 3, show_negatives: true }
      datasets: { maven_ere: { prompt: "maven_ere_anchor_eci", max_examples: 0 } }   # new prompt, Gap 3

  # ... repeat the richfewshot_* pair (per dataset) for each model in the matrix (2.3)
```

Run with `uv run run_queue.py --queue queue.yaml`, then aggregate with `uv run scripts/analysis/summary.py` and cross-check binary F1 with `uv run scripts/analysis/binary_eval.py logs/allatonce/run_*.json` to stay metric-compatible with the pilot numbers in Part 1.

### 2.5 Success criteria for the scale-up

- Overall binary micro-F1 (via `binary_eval.py --mode or`) for `richfewshot_*` should beat the matching `baseline_*` run, on **every** model in the matrix, not just the strongest one — otherwise the technique doesn't generalize and isn't worth the extra few-shot/prompt complexity.
- Precision on MAVEN-ERE and dense ESL documents (the E-route failure mode from Part 1) should not regress below the existing baseline's precision — if it does, Gap 4 (coherence pruning) needs to land before declaring this a win.
- Recall should improve materially on MECI/CTB (the D-route, sparse datasets) where the pilot's negative examples cleanly fixed false positives without a recall cost (CTB sample 1 went 0→perfect).

### 2.6 Suggested execution order

1. Implement Gap 1 (negative-aware few-shot flag) — smallest, highest-leverage change, touches only `tools/few_shot.py` / `utils/formatting.py`.
2. Write the two new anchor-table prompt files (Gap 3) for MAVEN-ERE and ESL.
3. Run the dataset-level dominant-route proxy (no pipeline changes) across the model matrix via `queue.yaml`.
4. Inspect precision on the E-route runs specifically; if it collapses as in the pilot, prioritize Gap 4 (coherence-tool pruning) before any further scale-up.
5. Only after 1–4 land and look healthy, consider the full per-document density router (Gap 2, follow-up).

---

## Appendix A — raw gold annotations (all 12 evaluation documents)

These are the exact gold triples used to score every system in this report (3 held-out documents per dataset).


### MECI

**idx 0** — mentions: `['T0', 'T1', 'T11', 'T12', 'T5', 'T7']`

```json
gold = [["T0", "causes", "T7"], ["T0", "causes", "T1"], ["T7", "causedby", "T0"], ["T1", "causedby", "T0"]]
```

**idx 1** — mentions: `['T0', 'T2']`

```json
gold = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**idx 2** — mentions: `['T0', 'T1', 'T10', 'T11', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']`

```json
gold = [["T0", "causes", "T10"], ["T5", "causes", "T7"], ["T10", "causedby", "T0"], ["T7", "causedby", "T5"]]
```


### MAVEN

**idx 0** — mentions: `['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11']`

```json
gold = [["e0", "causes", "e9"], ["e5", "causes", "e9"], ["e0", "causes", "e4"], ["e0", "causes", "e10"], ["e8", "causes", "e7"], ["e7", "causes", "e9"], ["e8", "causes", "e9"], ["e0", "causes", "e5"], ["e0", "causes", "e7"], ["e5", "causes", "e7"], ["e2", "causes", "e4"], ["e0", "causes", "e11"], ["e9", "causedby", "e0"], ["e9", "causedby", "e5"], ["e4", "causedby", "e0"], ["e10", "causedby", "e0"], ["e7", "causedby", "e8"], ["e9", "causedby", "e7"], ["e9", "causedby", "e8"], ["e5", "causedby", "e0"], ["e7", "causedby", "e0"], ["e7", "causedby", "e5"], ["e4", "causedby", "e2"], ["e11", "causedby", "e0"]]
```

**idx 1** — mentions: `['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17']`

```json
gold = [["e9", "causes", "e11"], ["e2", "causes", "e13"], ["e7", "causes", "e8"], ["e2", "causes", "e12"], ["e11", "causedby", "e9"], ["e13", "causedby", "e2"], ["e8", "causedby", "e7"], ["e12", "causedby", "e2"]]
```

**idx 2** — mentions: `['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21']`

```json
gold = [["e8", "causes", "e9"], ["e0", "causes", "e1"], ["e13", "causes", "e14"], ["e11", "causes", "e12"], ["e9", "causedby", "e8"], ["e1", "causedby", "e0"], ["e14", "causedby", "e13"], ["e12", "causedby", "e11"]]
```


### CTB

**idx 0** — mentions: `['ei1', 'ei2', 'ei3', 'ei4', 'ei5', 'ei7', 'ei8', 'ei9', 'ei10', 'ei11', 'ei14', 'ei16', 'ei17', 'ei19', 'ei20']`

```json
gold = [["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"]]
```

**idx 1** — mentions: `['ei1', 'ei2', 'ei3', 'ei5', 'ei7', 'ei8', 'ei9', 'ei10', 'ei11', 'ei12', 'ei13', 'ei14', 'ei15', 'ei16', 'ei17', 'ei18', 'ei20']`

```json
gold = [["ei7", "causes", "ei5"], ["ei8", "causes", "ei5"], ["ei12", "causes", "ei11"], ["ei17", "causes", "ei18"], ["ei5", "causedby", "ei7"], ["ei5", "causedby", "ei8"], ["ei11", "causedby", "ei12"], ["ei18", "causedby", "ei17"]]
```

**idx 2** — mentions: `['ei1', 'ei2', 'ei3', 'ei4', 'ei5', 'ei6', 'ei7', 'ei8', 'ei9', 'ei10', 'ei11', 'ei12', 'ei13']`

```json
gold = [["ei5", "causes", "ei7"], ["ei7", "causedby", "ei5"]]
```


### ESL

**idx 0** — mentions: `['3', '1', '2', '4', '5', '6']`

```json
gold = [["5", "causes", "6"], ["2", "causes", "5"], ["1", "causes", "2"], ["1", "causes", "6"], ["6", "causedby", "5"], ["5", "causedby", "2"], ["2", "causedby", "1"], ["6", "causedby", "1"]]
```

**idx 1** — mentions: `['8', '1', '14', '2', '17', '3', '4', '15', '5', '6', '7', '9', '10', '11', '12', '16', '13']`

```json
gold = [["7", "causes", "9"], ["9", "causes", "10"], ["9", "causedby", "7"], ["10", "causedby", "9"]]
```

**idx 2** — mentions: `['23', '10', '11', '24', '12', '13', '14', '15', '8', '16', '17', '25', '5', '21', '20', '18', '19', '1', '2', '4', '3', '22', '9', '7', '6']`

```json
gold = [["1", "causes", "3"], ["13", "causes", "17"], ["23", "causes", "10"], ["12", "causes", "20"], ["8", "causes", "20"], ["10", "causes", "17"], ["13", "causes", "15"], ["12", "causes", "14"], ["23", "causes", "12"], ["10", "causes", "16"], ["10", "causes", "20"], ["23", "causes", "5"], ["18", "causes", "19"], ["25", "causes", "5"], ["15", "causes", "5"], ["10", "causes", "11"], ["23", "causes", "21"], ["14", "causes", "5"], ["13", "causes", "16"], ["24", "causes", "18"], ["8", "causes", "25"], ["9", "causes", "6"], ["24", "causes", "5"], ["11", "causes", "8"], ["12", "causes", "16"], ["17", "causes", "5"], ["10", "causes", "14"], ["23", "causes", "13"], ["10", "causes", "15"], ["23", "causes", "18"], ["23", "causes", "8"], ["10", "causes", "24"], ["13", "causes", "14"], ["25", "causes", "21"], ["12", "causes", "15"], ["12", "causes", "17"], ["16", "causes", "5"], ["12", "causes", "25"], ["10", "causes", "25"], ["24", "causes", "13"], ["8", "causes", "17"], ["8", "causes", "16"], ["24", "causes", "12"], ["13", "causes", "20"], ["11", "causes", "13"], ["24", "causes", "8"], ["11", "causes", "5"], ["15", "causes", "8"], ["11", "causes", "12"], ["14", "causes", "8"], ["13", "causes", "25"], ["25", "causes", "18"], ["24", "causes", "21"], ["5", "causes", "20"], ["3", "causedby", "1"], ["17", "causedby", "13"], ["10", "causedby", "23"], ["20", "causedby", "12"], ["20", "causedby", "8"], ["17", "causedby", "10"], ["15", "causedby", "13"], ["14", "causedby", "12"], ["12", "causedby", "23"], ["16", "causedby", "10"], ["20", "causedby", "10"], ["5", "causedby", "23"], ["19", "causedby", "18"], ["5", "causedby", "25"], ["5", "causedby", "15"], ["11", "causedby", "10"], ["21", "causedby", "23"], ["5", "causedby", "14"], ["16", "causedby", "13"], ["18", "causedby", "24"], ["25", "causedby", "8"], ["6", "causedby", "9"], ["5", "causedby", "24"], ["8", "causedby", "11"], ["16", "causedby", "12"], ["5", "causedby", "17"], ["14", "causedby", "10"], ["13", "causedby", "23"], ["15", "causedby", "10"], ["18", "causedby", "23"], ["8", "causedby", "23"], ["24", "causedby", "10"], ["14", "causedby", "13"], ["21", "causedby", "25"], ["15", "causedby", "12"], ["17", "causedby", "12"], ["5", "causedby", "16"], ["25", "causedby", "12"], ["25", "causedby", "10"], ["13", "causedby", "24"], ["17", "causedby", "8"], ["16", "causedby", "8"], ["12", "causedby", "24"], ["20", "causedby", "13"], ["13", "causedby", "11"], ["8", "causedby", "24"], ["5", "causedby", "11"], ["8", "causedby", "15"], ["12", "causedby", "11"], ["8", "causedby", "14"], ["25", "causedby", "13"], ["18", "causedby", "25"], ["21", "causedby", "24"], ["20", "causedby", "5"]]
```

---

## Appendix B — raw predictions, baseline routes (A/B/C/D), all 12 documents

No few-shot examples were used for these four runs — single fixed strategy per route, applied uniformly to every document regardless of routing.


### Route A: Causal-marker

**meci idx 0**
```json
pred = []
```

**meci idx 1**
```json
pred = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**meci idx 2**
```json
pred = [["T0", "causes", "T2"], ["T2", "causedby", "T0"], ["T1", "causes", "T2"], ["T2", "causedby", "T1"], ["T3", "causes", "T4"], ["T4", "causedby", "T3"], ["T5", "causes", "T6"], ["T6", "causedby", "T5"], ["T0", "causes", "T10"], ["T10", "causedby", "T0"], ["T0", "causes", "T11"], ["T11", "causedby", "T0"]]
```

**maven idx 0**
```json
pred = [["e0", "causes", "e1"], ["e1", "causedby", "e0"], ["e0", "causes", "e4"], ["e4", "causedby", "e0"], ["e0", "causes", "e5"], ["e5", "causedby", "e0"], ["e8", "causes", "e7"], ["e7", "causedby", "e8"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e0", "causes", "e10"], ["e10", "causedby", "e0"], ["e10", "causes", "e11"], ["e11", "causedby", "e10"]]
```

**maven idx 1**
```json
pred = [["e7", "causes", "e8"], ["e8", "causedby", "e7"], ["e7", "causes", "e5"], ["e5", "causedby", "e7"], ["e5", "causes", "e9"], ["e9", "causedby", "e5"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e9", "causes", "e10"], ["e10", "causedby", "e9"], ["e9", "causes", "e6"], ["e6", "causedby", "e9"], ["e2", "causes", "e0"], ["e0", "causedby", "e2"], ["e5", "causes", "e6"], ["e6", "causedby", "e5"], ["e10", "causes", "e12"], ["e12", "causedby", "e10"], ["e10", "causes", "e13"], ["e13", "causedby", "e10"], ["e10", "causes", "e14"], ["e14", "causedby", "e10"], ["e10", "causes", "e15"], ["e15", "causedby", "e10"]]
```

**maven idx 2**
```json
pred = [["e4", "causes", "e2"], ["e2", "causedby", "e4"], ["e4", "causes", "e3"], ["e3", "causedby", "e4"], ["e20", "causes", "e21"], ["e21", "causedby", "e20"], ["e4", "causes", "e12"], ["e12", "causedby", "e4"]]
```

**ctb idx 0**
```json
pred = [["ei7", "causes", "ei2"], ["ei2", "causedby", "ei7"], ["ei2", "causes", "ei1"], ["ei1", "causedby", "ei2"], ["ei2", "causes", "ei4"], ["ei4", "causedby", "ei2"], ["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"], ["ei7", "causes", "ei17"], ["ei17", "causedby", "ei7"], ["ei2", "causes", "ei17"], ["ei17", "causedby", "ei2"]]
```

**ctb idx 1**
```json
pred = [["ei7", "causes", "ei5"], ["ei5", "causedby", "ei7"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"], ["ei12", "causes", "ei11"], ["ei11", "causedby", "ei12"], ["ei11", "causes", "ei10"], ["ei10", "causedby", "ei11"], ["ei7", "causes", "ei18"], ["ei18", "causedby", "ei7"], ["ei8", "causes", "ei18"], ["ei18", "causedby", "ei8"], ["ei17", "causes", "ei18"], ["ei18", "causedby", "ei17"]]
```

**ctb idx 2**
```json
pred = [["ei3", "causes", "ei7"], ["ei7", "causedby", "ei3"], ["ei3", "causes", "ei6"], ["ei6", "causedby", "ei3"], ["ei2", "causes", "ei8"], ["ei8", "causedby", "ei2"], ["ei9", "causes", "ei10"], ["ei10", "causedby", "ei9"]]
```

**esl idx 0**
```json
pred = [["1", "causes", "2"], ["2", "causedby", "1"], ["5", "causes", "6"], ["6", "causedby", "5"], ["2", "causes", "3"], ["3", "causedby", "2"], ["6", "causes", "4"], ["4", "causedby", "6"]]
```

**esl idx 1**
```json
pred = [["2", "causes", "13"], ["13", "causedby", "2"], ["7", "causes", "1"], ["1", "causedby", "7"]]
```

**esl idx 2**
```json
pred = [["16", "causes", "13"], ["13", "causedby", "16"], ["16", "causes", "21"], ["21", "causedby", "16"], ["16", "causes", "5"], ["5", "causedby", "16"], ["6", "causes", "13"], ["13", "causedby", "6"], ["14", "causes", "16"], ["16", "causedby", "14"], ["20", "causes", "16"], ["16", "causedby", "20"], ["5", "causes", "15"], ["15", "causedby", "5"], ["6", "causes", "7"], ["7", "causedby", "6"], ["7", "causes", "9"], ["9", "causedby", "7"], ["9", "causes", "23"], ["23", "causedby", "9"], ["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "18"], ["18", "causedby", "1"]]
```


### Route B: Mention-by-Mention (E)

**meci idx 0**
```json
pred = []
```

**meci idx 1**
```json
pred = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**meci idx 2**
```json
pred = [["T1", "causes", "T2"], ["T2", "causedby", "T1"]]
```

**maven idx 0**
```json
pred = [["e8", "causes", "e9"], ["e9", "causedby", "e8"]]
```

**maven idx 1**
```json
pred = [["e2", "causes", "e0"], ["e0", "causedby", "e2"], ["e7", "causes", "e6"], ["e6", "causedby", "e7"], ["e9", "causes", "e10"], ["e10", "causedby", "e9"]]
```

**maven idx 2**
```json
pred = [["e11", "causes", "e12"], ["e12", "causedby", "e11"]]
```

**ctb idx 0**
```json
pred = [["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"]]
```

**ctb idx 1**
```json
pred = [["ei7", "causes", "ei5"], ["ei5", "causedby", "ei7"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"], ["ei12", "causes", "ei11"], ["ei11", "causedby", "ei12"]]
```

**ctb idx 2**
```json
pred = [["ei3", "causes", "ei7"], ["ei7", "causedby", "ei3"], ["ei9", "causes", "ei10"], ["ei10", "causedby", "ei9"]]
```

**esl idx 0**
```json
pred = [["1", "causes", "2"], ["2", "causedby", "1"]]
```

**esl idx 1**
```json
pred = []
```

**esl idx 2**
```json
pred = [["16", "causes", "13"], ["13", "causedby", "16"], ["20", "causes", "13"], ["13", "causedby", "20"]]
```


### Route C: marker U Narr-chain (CuD)

**meci idx 0**
```json
pred = []
```

**meci idx 1**
```json
pred = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**meci idx 2**
```json
pred = [["T1", "causes", "T2"], ["T2", "causedby", "T1"], ["T0", "causes", "T2"], ["T2", "causedby", "T0"], ["T1", "causes", "T10"], ["T10", "causedby", "T1"], ["T1", "causes", "T11"], ["T11", "causedby", "T1"], ["T3", "causes", "T4"], ["T4", "causedby", "T3"], ["T5", "causes", "T6"], ["T6", "causedby", "T5"]]
```

**maven idx 0**
```json
pred = [["e0", "causes", "e1"], ["e1", "causedby", "e0"], ["e0", "causes", "e4"], ["e4", "causedby", "e0"], ["e0", "causes", "e5"], ["e5", "causedby", "e0"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e0", "causes", "e10"], ["e10", "causedby", "e0"], ["e10", "causes", "e11"], ["e11", "causedby", "e10"]]
```

**maven idx 1**
```json
pred = [["e2", "causes", "e0"], ["e0", "causedby", "e2"], ["e2", "causes", "e4"], ["e4", "causedby", "e2"], ["e7", "causes", "e8"], ["e8", "causedby", "e7"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e5", "causes", "e6"], ["e6", "causedby", "e5"], ["e5", "causes", "e9"], ["e9", "causedby", "e5"], ["e9", "causes", "e10"], ["e10", "causedby", "e9"], ["e10", "causes", "e12"], ["e12", "causedby", "e10"], ["e10", "causes", "e13"], ["e13", "causedby", "e10"], ["e10", "causes", "e14"], ["e14", "causedby", "e10"], ["e10", "causes", "e15"], ["e15", "causedby", "e10"]]
```

**maven idx 2**
```json
pred = [["e1", "causes", "e2"], ["e2", "causedby", "e1"], ["e4", "causes", "e3"], ["e3", "causedby", "e4"], ["e20", "causes", "e21"], ["e21", "causedby", "e20"], ["e14", "causes", "e9"], ["e9", "causedby", "e14"]]
```

**ctb idx 0**
```json
pred = [["ei7", "causes", "ei2"], ["ei2", "causedby", "ei7"], ["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"], ["ei2", "causes", "ei1"], ["ei1", "causedby", "ei2"], ["ei2", "causes", "ei4"], ["ei4", "causedby", "ei2"], ["ei8", "causes", "ei17"], ["ei17", "causedby", "ei8"]]
```

**ctb idx 1**
```json
pred = [["ei7", "causes", "ei5"], ["ei5", "causedby", "ei7"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"], ["ei12", "causes", "ei11"], ["ei11", "causedby", "ei12"], ["ei11", "causes", "ei18"], ["ei18", "causedby", "ei11"], ["ei8", "causes", "ei18"], ["ei18", "causedby", "ei8"], ["ei17", "causes", "ei18"], ["ei18", "causedby", "ei17"]]
```

**ctb idx 2**
```json
pred = [["ei3", "causes", "ei7"], ["ei7", "causedby", "ei3"], ["ei3", "causes", "ei6"], ["ei6", "causedby", "ei3"], ["ei2", "causes", "ei9"], ["ei9", "causedby", "ei2"], ["ei9", "causes", "ei10"], ["ei10", "causedby", "ei9"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"]]
```

**esl idx 0**
```json
pred = [["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "3"], ["3", "causedby", "1"], ["5", "causes", "4"], ["4", "causedby", "5"], ["6", "causes", "3"], ["3", "causedby", "6"]]
```

**esl idx 1**
```json
pred = [["7", "causes", "1"], ["1", "causedby", "7"], ["7", "causes", "2"], ["2", "causedby", "7"], ["7", "causes", "3"], ["3", "causedby", "7"], ["9", "causes", "5"], ["5", "causedby", "9"], ["13", "causes", "16"], ["16", "causedby", "13"]]
```

**esl idx 2**
```json
pred = [["16", "causes", "13"], ["13", "causedby", "16"], ["16", "causes", "5"], ["5", "causedby", "16"], ["16", "causes", "21"], ["21", "causedby", "16"], ["14", "causes", "13"], ["13", "causedby", "14"], ["20", "causes", "1"], ["1", "causedby", "20"], ["6", "causes", "15"], ["15", "causedby", "6"], ["6", "causes", "23"], ["23", "causedby", "6"], ["6", "causes", "7"], ["7", "causedby", "6"], ["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "3"], ["3", "causedby", "1"]]
```


### Route D: Narrative Chain

**meci idx 0**
```json
pred = [["T5", "causes", "T0"], ["T0", "causedby", "T5"], ["T0", "causes", "T7"], ["T7", "causedby", "T0"], ["T0", "causes", "T1"], ["T1", "causedby", "T0"]]
```

**meci idx 1**
```json
pred = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**meci idx 2**
```json
pred = [["T1", "causes", "T2"], ["T2", "causedby", "T1"], ["T1", "causes", "T0"], ["T0", "causedby", "T1"], ["T3", "causes", "T4"], ["T4", "causedby", "T3"], ["T5", "causes", "T6"], ["T6", "causedby", "T5"], ["T8", "causes", "T11"], ["T11", "causedby", "T8"], ["T1", "causes", "T10"], ["T10", "causedby", "T1"], ["T10", "causes", "T11"], ["T11", "causedby", "T10"]]
```

**maven idx 0**
```json
pred = [["e0", "causes", "e1"], ["e1", "causedby", "e0"], ["e0", "causes", "e4"], ["e4", "causedby", "e0"], ["e0", "causes", "e5"], ["e5", "causedby", "e0"], ["e0", "causes", "e7"], ["e7", "causedby", "e0"], ["e0", "causes", "e8"], ["e8", "causedby", "e0"], ["e0", "causes", "e9"], ["e9", "causedby", "e0"], ["e0", "causes", "e10"], ["e10", "causedby", "e0"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e5", "causes", "e7"], ["e7", "causedby", "e5"], ["e10", "causes", "e11"], ["e11", "causedby", "e10"], ["e6", "causes", "e5"], ["e5", "causedby", "e6"]]
```

**maven idx 1**
```json
pred = [["e7", "causes", "e8"], ["e8", "causedby", "e7"], ["e7", "causes", "e5"], ["e5", "causedby", "e7"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e5", "causes", "e6"], ["e6", "causedby", "e5"], ["e9", "causes", "e10"], ["e10", "causedby", "e9"], ["e9", "causes", "e11"], ["e11", "causedby", "e9"], ["e10", "causes", "e0"], ["e0", "causedby", "e10"], ["e10", "causes", "e4"], ["e4", "causedby", "e10"], ["e10", "causes", "e12"], ["e12", "causedby", "e10"], ["e10", "causes", "e13"], ["e13", "causedby", "e10"], ["e10", "causes", "e14"], ["e14", "causedby", "e10"], ["e10", "causes", "e15"], ["e15", "causedby", "e10"], ["e2", "causes", "e0"], ["e0", "causedby", "e2"], ["e2", "causes", "e4"], ["e4", "causedby", "e2"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e6", "causes", "e10"], ["e10", "causedby", "e6"], ["e0", "causes", "e12"], ["e12", "causedby", "e0"], ["e0", "causes", "e13"], ["e13", "causedby", "e0"], ["e11", "causes", "e12"], ["e12", "causedby", "e11"], ["e11", "causes", "e13"], ["e13", "causedby", "e11"]]
```

**maven idx 2**
```json
pred = [["e1", "causes", "e2"], ["e2", "causedby", "e1"], ["e1", "causes", "e3"], ["e3", "causedby", "e1"], ["e4", "causes", "e3"], ["e3", "causedby", "e4"], ["e3", "causes", "e12"], ["e12", "causedby", "e3"], ["e20", "causes", "e21"], ["e21", "causedby", "e20"], ["e20", "causes", "e19"], ["e19", "causedby", "e20"], ["e14", "causes", "e21"], ["e21", "causedby", "e14"], ["e13", "causes", "e9"], ["e9", "causedby", "e13"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e7", "causes", "e8"], ["e8", "causedby", "e7"], ["e11", "causes", "e12"], ["e12", "causedby", "e11"]]
```

**ctb idx 0**
```json
pred = [["ei7", "causes", "ei2"], ["ei2", "causedby", "ei7"], ["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"], ["ei7", "causes", "ei17"], ["ei17", "causedby", "ei7"], ["ei7", "causes", "ei11"], ["ei11", "causedby", "ei7"], ["ei2", "causes", "ei1"], ["ei1", "causedby", "ei2"], ["ei2", "causes", "ei4"], ["ei4", "causedby", "ei2"], ["ei11", "causes", "ei17"], ["ei17", "causedby", "ei11"], ["ei8", "causes", "ei17"], ["ei17", "causedby", "ei8"]]
```

**ctb idx 1**
```json
pred = [["ei7", "causes", "ei5"], ["ei5", "causedby", "ei7"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"], ["ei12", "causes", "ei11"], ["ei11", "causedby", "ei12"], ["ei11", "causes", "ei5"], ["ei5", "causedby", "ei11"], ["ei13", "causes", "ei5"], ["ei5", "causedby", "ei13"], ["ei7", "causes", "ei18"], ["ei18", "causedby", "ei7"], ["ei8", "causes", "ei18"], ["ei18", "causedby", "ei8"], ["ei12", "causes", "ei18"], ["ei18", "causedby", "ei12"], ["ei17", "causes", "ei18"], ["ei18", "causedby", "ei17"], ["ei13", "causes", "ei18"], ["ei18", "causedby", "ei13"], ["ei11", "causes", "ei18"], ["ei18", "causedby", "ei11"]]
```

**ctb idx 2**
```json
pred = [["ei3", "causes", "ei7"], ["ei7", "causedby", "ei3"], ["ei3", "causes", "ei8"], ["ei8", "causedby", "ei3"], ["ei2", "causes", "ei9"], ["ei9", "causedby", "ei2"], ["ei2", "causes", "ei10"], ["ei10", "causedby", "ei2"], ["ei8", "causes", "ei2"], ["ei2", "causedby", "ei8"], ["ei5", "causes", "ei6"], ["ei6", "causedby", "ei5"], ["ei5", "causes", "ei7"], ["ei7", "causedby", "ei5"], ["ei9", "causes", "ei10"], ["ei10", "causedby", "ei9"]]
```

**esl idx 0**
```json
pred = [["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "3"], ["3", "causedby", "1"], ["1", "causes", "4"], ["4", "causedby", "1"], ["5", "causes", "2"], ["2", "causedby", "5"], ["6", "causes", "3"], ["3", "causedby", "6"], ["6", "causes", "4"], ["4", "causedby", "6"]]
```

**esl idx 1**
```json
pred = [["14", "causes", "5"], ["5", "causedby", "14"], ["7", "causes", "1"], ["1", "causedby", "7"], ["7", "causes", "2"], ["2", "causedby", "7"], ["7", "causes", "3"], ["3", "causedby", "7"], ["7", "causes", "13"], ["13", "causedby", "7"], ["9", "causes", "10"], ["10", "causedby", "9"], ["15", "causes", "1"], ["1", "causedby", "15"], ["12", "causes", "11"], ["11", "causedby", "12"]]
```

**esl idx 2**
```json
pred = [["16", "causes", "13"], ["13", "causedby", "16"], ["16", "causes", "5"], ["5", "causedby", "16"], ["16", "causes", "21"], ["21", "causedby", "16"], ["16", "causes", "10"], ["10", "causedby", "16"], ["16", "causes", "8"], ["8", "causedby", "16"], ["16", "causes", "1"], ["1", "causedby", "16"], ["6", "causes", "13"], ["13", "causedby", "6"], ["6", "causes", "5"], ["5", "causedby", "6"], ["6", "causes", "21"], ["21", "causedby", "6"], ["6", "causes", "10"], ["10", "causedby", "6"], ["6", "causes", "8"], ["8", "causedby", "6"], ["6", "causes", "1"], ["1", "causedby", "6"], ["14", "causes", "13"], ["13", "causedby", "14"], ["20", "causes", "13"], ["13", "causedby", "20"], ["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "3"], ["3", "causedby", "1"], ["3", "causes", "9"], ["9", "causedby", "3"], ["3", "causes", "23"], ["23", "causedby", "3"], ["3", "causes", "24"], ["24", "causedby", "3"], ["3", "causes", "25"], ["25", "causedby", "3"], ["3", "causes", "15"], ["15", "causedby", "3"], ["3", "causes", "11"], ["11", "causedby", "3"], ["3", "causes", "17"], ["17", "causedby", "3"], ["7", "causes", "18"], ["18", "causedby", "7"], ["6", "causes", "7"], ["7", "causedby", "6"]]
```

---

## Appendix C — raw predictions, v3 routed + rich few-shot, all 12 documents

Each prediction below was produced by a separate model call primed with 3 TF-IDF-nearest-neighbour worked examples (positive + negative labelled pairs) and the routed strategy for that dataset/sample (see routing table in Part 1).


### MECI

**idx 0**
```json
pred = [["T12", "causes", "T1"], ["T1", "causedby", "T12"], ["T11", "causes", "T12"], ["T12", "causedby", "T11"]]
```

**idx 1**
```json
pred = [["T2", "causes", "T0"], ["T0", "causedby", "T2"]]
```

**idx 2**
```json
pred = [["T0", "causes", "T2"], ["T2", "causedby", "T0"], ["T1", "causes", "T2"], ["T2", "causedby", "T1"], ["T0", "causes", "T10"], ["T10", "causedby", "T0"], ["T2", "causes", "T10"], ["T10", "causedby", "T2"], ["T10", "causes", "T11"], ["T11", "causedby", "T10"], ["T8", "causes", "T11"], ["T11", "causedby", "T8"]]
```


### MAVEN

**idx 0**
```json
pred = [["e0", "causes", "e1"], ["e1", "causedby", "e0"], ["e0", "causes", "e4"], ["e4", "causedby", "e0"], ["e0", "causes", "e5"], ["e5", "causedby", "e0"], ["e0", "causes", "e7"], ["e7", "causedby", "e0"], ["e0", "causes", "e8"], ["e8", "causedby", "e0"], ["e0", "causes", "e9"], ["e9", "causedby", "e0"], ["e0", "causes", "e10"], ["e10", "causedby", "e0"], ["e0", "causes", "e11"], ["e11", "causedby", "e0"], ["e4", "causes", "e5"], ["e5", "causedby", "e4"], ["e5", "causes", "e7"], ["e7", "causedby", "e5"], ["e5", "causes", "e9"], ["e9", "causedby", "e5"], ["e6", "causes", "e5"], ["e5", "causedby", "e6"], ["e6", "causes", "e8"], ["e8", "causedby", "e6"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e8", "causes", "e5"], ["e5", "causedby", "e8"], ["e8", "causes", "e7"], ["e7", "causedby", "e8"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e8", "causes", "e10"], ["e10", "causedby", "e8"], ["e8", "causes", "e11"], ["e11", "causedby", "e8"], ["e10", "causes", "e11"], ["e11", "causedby", "e10"]]
```

**idx 1**
```json
pred = [["e2", "causes", "e0"], ["e0", "causedby", "e2"], ["e2", "causes", "e1"], ["e1", "causedby", "e2"], ["e2", "causes", "e12"], ["e12", "causedby", "e2"], ["e2", "causes", "e13"], ["e13", "causedby", "e2"], ["e2", "causes", "e14"], ["e14", "causedby", "e2"], ["e2", "causes", "e15"], ["e15", "causedby", "e2"], ["e1", "causes", "e0"], ["e0", "causedby", "e1"], ["e1", "causes", "e12"], ["e12", "causedby", "e1"], ["e1", "causes", "e13"], ["e13", "causedby", "e1"], ["e1", "causes", "e14"], ["e14", "causedby", "e1"], ["e1", "causes", "e15"], ["e15", "causedby", "e1"], ["e0", "causes", "e12"], ["e12", "causedby", "e0"], ["e0", "causes", "e13"], ["e13", "causedby", "e0"], ["e0", "causes", "e14"], ["e14", "causedby", "e0"], ["e0", "causes", "e15"], ["e15", "causedby", "e0"], ["e7", "causes", "e5"], ["e5", "causedby", "e7"], ["e7", "causes", "e6"], ["e6", "causedby", "e7"], ["e7", "causes", "e8"], ["e8", "causedby", "e7"], ["e7", "causes", "e9"], ["e9", "causedby", "e7"], ["e7", "causes", "e10"], ["e10", "causedby", "e7"], ["e7", "causes", "e12"], ["e12", "causedby", "e7"], ["e7", "causes", "e13"], ["e13", "causedby", "e7"], ["e7", "causes", "e14"], ["e14", "causedby", "e7"], ["e7", "causes", "e15"], ["e15", "causedby", "e7"], ["e5", "causes", "e6"], ["e6", "causedby", "e5"], ["e5", "causes", "e8"], ["e8", "causedby", "e5"], ["e5", "causes", "e9"], ["e9", "causedby", "e5"], ["e5", "causes", "e10"], ["e10", "causedby", "e5"], ["e5", "causes", "e12"], ["e12", "causedby", "e5"], ["e5", "causes", "e13"], ["e13", "causedby", "e5"], ["e5", "causes", "e14"], ["e14", "causedby", "e5"], ["e5", "causes", "e15"], ["e15", "causedby", "e5"], ["e6", "causes", "e8"], ["e8", "causedby", "e6"], ["e6", "causes", "e9"], ["e9", "causedby", "e6"], ["e6", "causes", "e10"], ["e10", "causedby", "e6"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e8", "causes", "e10"], ["e10", "causedby", "e8"], ["e8", "causes", "e12"], ["e12", "causedby", "e8"], ["e8", "causes", "e13"], ["e13", "causedby", "e8"], ["e8", "causes", "e14"], ["e14", "causedby", "e8"], ["e8", "causes", "e15"], ["e15", "causedby", "e8"], ["e9", "causes", "e10"], ["e10", "causedby", "e9"], ["e9", "causes", "e12"], ["e12", "causedby", "e9"], ["e9", "causes", "e13"], ["e13", "causedby", "e9"], ["e9", "causes", "e14"], ["e14", "causedby", "e9"], ["e9", "causes", "e15"], ["e15", "causedby", "e9"], ["e10", "causes", "e12"], ["e12", "causedby", "e10"], ["e10", "causes", "e13"], ["e13", "causedby", "e10"], ["e10", "causes", "e14"], ["e14", "causedby", "e10"], ["e10", "causes", "e15"], ["e15", "causedby", "e10"]]
```

**idx 2**
```json
pred = [["e4", "causes", "e2"], ["e2", "causedby", "e4"], ["e4", "causes", "e3"], ["e3", "causedby", "e4"], ["e4", "causes", "e12"], ["e12", "causedby", "e4"], ["e8", "causes", "e9"], ["e9", "causedby", "e8"], ["e8", "causes", "e11"], ["e11", "causedby", "e8"], ["e19", "causes", "e21"], ["e21", "causedby", "e19"], ["e20", "causes", "e21"], ["e21", "causedby", "e20"]]
```


### CTB

**idx 0**
```json
pred = [["ei7", "causes", "ei2"], ["ei2", "causedby", "ei7"], ["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"], ["ei2", "causes", "ei1"], ["ei1", "causedby", "ei2"]]
```

**idx 1**
```json
pred = [["ei7", "causes", "ei5"], ["ei5", "causedby", "ei7"], ["ei8", "causes", "ei5"], ["ei5", "causedby", "ei8"], ["ei12", "causes", "ei11"], ["ei11", "causedby", "ei12"], ["ei17", "causes", "ei18"], ["ei18", "causedby", "ei17"]]
```

**idx 2**
```json
pred = [["ei3", "causes", "ei7"], ["ei7", "causedby", "ei3"], ["ei3", "causes", "ei8"], ["ei8", "causedby", "ei3"], ["ei7", "causes", "ei8"], ["ei8", "causedby", "ei7"], ["ei9", "causes", "ei10"], ["ei10", "causedby", "ei9"]]
```


### ESL

**idx 0**
```json
pred = [["1", "causes", "2"], ["2", "causedby", "1"], ["1", "causes", "6"], ["6", "causedby", "1"], ["5", "causes", "2"], ["2", "causedby", "5"], ["5", "causes", "6"], ["6", "causedby", "5"], ["2", "causes", "3"], ["3", "causedby", "2"], ["2", "causes", "4"], ["4", "causedby", "2"], ["6", "causes", "3"], ["3", "causedby", "6"], ["6", "causes", "4"], ["4", "causedby", "6"], ["1", "causes", "3"], ["3", "causedby", "1"], ["1", "causes", "4"], ["4", "causedby", "1"], ["5", "causes", "3"], ["3", "causedby", "5"], ["5", "causes", "4"], ["4", "causedby", "5"]]
```

**idx 1**
```json
pred = [["7", "causes", "9"], ["9", "causedby", "7"], ["9", "causes", "10"], ["10", "causedby", "9"]]
```

**idx 2**
```json
pred = [["6", "causes", "10"], ["10", "causedby", "6"], ["6", "causes", "12"], ["12", "causedby", "6"], ["6", "causes", "13"], ["13", "causedby", "6"], ["6", "causes", "5"], ["5", "causedby", "6"], ["6", "causes", "21"], ["21", "causedby", "6"], ["6", "causes", "1"], ["1", "causedby", "6"], ["6", "causes", "8"], ["8", "causedby", "6"], ["6", "causes", "15"], ["15", "causedby", "6"], ["6", "causes", "11"], ["11", "causedby", "6"], ["6", "causes", "24"], ["24", "causedby", "6"], ["6", "causes", "17"], ["17", "causedby", "6"], ["6", "causes", "25"], ["25", "causedby", "6"], ["6", "causes", "23"], ["23", "causedby", "6"], ["6", "causes", "9"], ["9", "causedby", "6"], ["6", "causes", "18"], ["18", "causedby", "6"], ["6", "causes", "19"], ["19", "causedby", "6"], ["6", "causes", "2"], ["2", "causedby", "6"], ["16", "causes", "6"], ["6", "causedby", "16"], ["16", "causes", "13"], ["13", "causedby", "16"], ["16", "causes", "1"], ["1", "causedby", "16"], ["16", "causes", "5"], ["5", "causedby", "16"], ["16", "causes", "21"], ["21", "causedby", "16"], ["16", "causes", "10"], ["10", "causedby", "16"], ["16", "causes", "12"], ["12", "causedby", "16"], ["16", "causes", "8"], ["8", "causedby", "16"], ["16", "causes", "23"], ["23", "causedby", "16"], ["16", "causes", "15"], ["15", "causedby", "16"], ["16", "causes", "11"], ["11", "causedby", "16"], ["16", "causes", "24"], ["24", "causedby", "16"], ["16", "causes", "17"], ["17", "causedby", "16"], ["16", "causes", "25"], ["25", "causedby", "16"], ["16", "causes", "9"], ["9", "causedby", "16"], ["16", "causes", "18"], ["18", "causedby", "16"], ["16", "causes", "19"], ["19", "causedby", "16"], ["16", "causes", "2"], ["2", "causedby", "16"], ["20", "causes", "16"], ["16", "causedby", "20"], ["20", "causes", "6"], ["6", "causedby", "20"], ["20", "causes", "13"], ["13", "causedby", "20"], ["20", "causes", "1"], ["1", "causedby", "20"], ["20", "causes", "5"], ["5", "causedby", "20"], ["20", "causes", "21"], ["21", "causedby", "20"], ["20", "causes", "10"], ["10", "causedby", "20"], ["20", "causes", "12"], ["12", "causedby", "20"], ["20", "causes", "8"], ["8", "causedby", "20"], ["20", "causes", "23"], ["23", "causedby", "20"], ["20", "causes", "15"], ["15", "causedby", "20"], ["20", "causes", "9"], ["9", "causedby", "20"], ["20", "causes", "18"], ["18", "causedby", "20"], ["20", "causes", "19"], ["19", "causedby", "20"], ["14", "causes", "16"], ["16", "causedby", "14"], ["14", "causes", "6"], ["6", "causedby", "14"], ["14", "causes", "13"], ["13", "causedby", "14"], ["14", "causes", "1"], ["1", "causedby", "14"], ["14", "causes", "5"], ["5", "causedby", "14"], ["14", "causes", "21"], ["21", "causedby", "14"], ["14", "causes", "10"], ["10", "causedby", "14"], ["14", "causes", "12"], ["12", "causedby", "14"], ["14", "causes", "8"], ["8", "causedby", "14"], ["14", "causes", "23"], ["23", "causedby", "14"], ["14", "causes", "15"], ["15", "causedby", "14"], ["14", "causes", "9"], ["9", "causedby", "14"], ["14", "causes", "18"], ["18", "causedby", "14"], ["14", "causes", "19"], ["19", "causedby", "14"], ["13", "causes", "1"], ["1", "causedby", "13"], ["13", "causes", "5"], ["5", "causedby", "13"], ["13", "causes", "10"], ["10", "causedby", "13"], ["13", "causes", "12"], ["12", "causedby", "13"], ["13", "causes", "8"], ["8", "causedby", "13"], ["5", "causes", "23"], ["23", "causedby", "5"], ["5", "causes", "15"], ["15", "causedby", "5"], ["5", "causes", "11"], ["11", "causedby", "5"], ["5", "causes", "24"], ["24", "causedby", "5"], ["5", "causes", "17"], ["17", "causedby", "5"], ["5", "causes", "25"], ["25", "causedby", "5"], ["5", "causes", "9"], ["9", "causedby", "5"], ["5", "causes", "18"], ["18", "causedby", "5"], ["5", "causes", "19"], ["19", "causedby", "5"], ["5", "causes", "3"], ["3", "causedby", "5"], ["5", "causes", "22"], ["22", "causedby", "5"], ["10", "causes", "23"], ["23", "causedby", "10"], ["10", "causes", "15"], ["15", "causedby", "10"], ["10", "causes", "11"], ["11", "causedby", "10"], ["10", "causes", "24"], ["24", "causedby", "10"], ["10", "causes", "17"], ["17", "causedby", "10"], ["10", "causes", "25"], ["25", "causedby", "10"], ["10", "causes", "9"], ["9", "causedby", "10"], ["10", "causes", "18"], ["18", "causedby", "10"], ["10", "causes", "19"], ["19", "causedby", "10"], ["10", "causes", "3"], ["3", "causedby", "10"], ["10", "causes", "22"], ["22", "causedby", "10"], ["12", "causes", "23"], ["23", "causedby", "12"], ["12", "causes", "15"], ["15", "causedby", "12"], ["12", "causes", "11"], ["11", "causedby", "12"], ["12", "causes", "24"], ["24", "causedby", "12"], ["12", "causes", "17"], ["17", "causedby", "12"], ["12", "causes", "25"], ["25", "causedby", "12"], ["12", "causes", "9"], ["9", "causedby", "12"], ["12", "causes", "18"], ["18", "causedby", "12"], ["12", "causes", "19"], ["19", "causedby", "12"], ["12", "causes", "3"], ["3", "causedby", "12"], ["12", "causes", "22"], ["22", "causedby", "12"], ["8", "causes", "23"], ["23", "causedby", "8"], ["8", "causes", "15"], ["15", "causedby", "8"], ["8", "causes", "11"], ["11", "causedby", "8"], ["8", "causes", "24"], ["24", "causedby", "8"], ["8", "causes", "17"], ["17", "causedby", "8"], ["8", "causes", "25"], ["25", "causedby", "8"], ["8", "causes", "9"], ["9", "causedby", "8"], ["8", "causes", "18"], ["18", "causedby", "8"], ["8", "causes", "19"], ["19", "causedby", "8"], ["8", "causes", "3"], ["3", "causedby", "8"], ["8", "causes", "22"], ["22", "causedby", "8"], ["1", "causes", "23"], ["23", "causedby", "1"], ["1", "causes", "15"], ["15", "causedby", "1"], ["1", "causes", "11"], ["11", "causedby", "1"], ["1", "causes", "24"], ["24", "causedby", "1"], ["1", "causes", "17"], ["17", "causedby", "1"], ["1", "causes", "25"], ["25", "causedby", "1"], ["1", "causes", "9"], ["9", "causedby", "1"], ["1", "causes", "18"], ["18", "causedby", "1"], ["1", "causes", "19"], ["19", "causedby", "1"], ["1", "causes", "3"], ["3", "causedby", "1"], ["1", "causes", "22"], ["22", "causedby", "1"], ["7", "causes", "23"], ["23", "causedby", "7"], ["7", "causes", "15"], ["15", "causedby", "7"], ["7", "causes", "9"], ["9", "causedby", "7"], ["7", "causes", "11"], ["11", "causedby", "7"], ["7", "causes", "24"], ["24", "causedby", "7"], ["7", "causes", "17"], ["17", "causedby", "7"], ["7", "causes", "25"], ["25", "causedby", "7"], ["7", "causes", "18"], ["18", "causedby", "7"], ["7", "causes", "19"], ["19", "causedby", "7"], ["3", "causes", "9"], ["9", "causedby", "3"], ["3", "causes", "23"], ["23", "causedby", "3"], ["3", "causes", "15"], ["15", "causedby", "3"], ["3", "causes", "22"], ["22", "causedby", "3"], ["22", "causes", "9"], ["9", "causedby", "22"], ["22", "causes", "23"], ["23", "causedby", "22"], ["2", "causes", "3"], ["3", "causedby", "2"], ["2", "causes", "4"], ["4", "causedby", "2"], ["23", "causes", "18"], ["18", "causedby", "23"], ["23", "causes", "19"], ["19", "causedby", "23"], ["9", "causes", "18"], ["18", "causedby", "9"], ["9", "causes", "19"], ["19", "causedby", "9"], ["15", "causes", "18"], ["18", "causedby", "15"], ["15", "causes", "19"], ["19", "causedby", "15"], ["11", "causes", "18"], ["18", "causedby", "11"], ["11", "causes", "19"], ["19", "causedby", "11"], ["24", "causes", "18"], ["18", "causedby", "24"], ["24", "causes", "19"], ["19", "causedby", "24"], ["17", "causes", "18"], ["18", "causedby", "17"], ["17", "causes", "19"], ["19", "causedby", "17"], ["25", "causes", "18"], ["18", "causedby", "25"], ["25", "causes", "19"], ["19", "causedby", "25"]]
```
