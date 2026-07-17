# Ablation catalogue — what is already in `mlflow.db`

_Generated 2026-07-15 from 208 FINISHED runs with logged params._

## How to read this

A group below is a set of runs where **every knob is identical except the one named**, so the
comparison is clean. Derived params (`prompt.system`/`user_template`, which follow from
`dataset.prompt`; `dataset.name`/`repo_id`/`split`, which follow from `active_dataset`) and infra
params (concurrency, timeout, retries, tracing) are excluded from the signature — they cannot
change predictions.

**Noise floor.** 31 groups in the DB have *byte-identical* configs and were simply re-run.
Their F1 spread is median **0.024**, p75 **0.090**, max 0.292. An independent paired
bootstrap over documents on the ESL 50-doc runs gives a 95% CI of about **±0.05 F1**. So:

- Δ < 0.02 → **noise**, tells you nothing
- Δ 0.02–0.09 → **suggestive only**, within rerun spread
- Δ > 0.09 → **worth believing** (still only from n=50 unless noted)

Metric per group is the best one all its runs logged: `intraF1` = binary_intra_f1, `binF1` =
binary_f1, `microF1` = micro_f1. Runs that scored exactly 0.0 (degenerate/broken) are dropped.

Values are means when a group has replicates.

---

## Summary — coverage by axis

| axis | groups | Δ>p75 | datasets covered |
|---|---:|---:|---|
| `model.default_model_id` | 16 | 9 | causal_timebank, event_story_line, maven_ere, meci |
| `dataset.max_examples` | 10 | 5 | event_story_line, maven_ere, meci |
| `dataset.prompt` | 17 | 5 | causal_timebank, event_story_line, maven_ere, meci |
| `few_shot.enabled` | 14 | 4 | causal_timebank, event_story_line, maven_ere, meci |
| `data.novote_norel` | 1 | 1 | meci |
| `experiment.resampling.enabled` | 11 | 1 | causal_timebank, event_story_line, maven_ere, meci |
| `few_shot.cot_generation.enabled` | 5 | 1 | causal_timebank, meci |
| `few_shot.systematic` | 2 | 1 | causal_timebank, meci |
| `data.binary_undirected` | 2 | 0 | causal_timebank, meci |
| `experiment.resampling.distinct_fewshots` | 1 | 0 | maven_ere |
| `experiment.tools` | 1 | 0 | maven_ere |
| `few_shot.cot_generation.num_steps` | 1 | 0 | maven_ere |
| `few_shot.n_examples` | 1 | 0 | maven_ere |
| `few_shot.pool_size` | 8 | 0 | causal_timebank, event_story_line, maven_ere, meci |
| `few_shot.selection` | 2 | 0 | maven_ere, meci |

**Never cleanly ablated** (no group isolates them): `active_dataset`, `data.vote_to_binary`, `dataset.constrain_to_pair_list`, `dataset.kfold.enabled`, `dataset.kfold.n_folds`, `experiment.enable_tools`, `experiment.relation_budget.enabled`, `experiment.resampling.n_runs`, `few_shot.cot_generation.same_relation_budget`

---

## 1-knob ablations, by axis

### `model.default_model_id`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, ?, causal_timebank_explicit_marker | binF1 | `deepseek/deepseek-v3.2:nit`→**0.383** · `openai/gpt-5.4-mini`→0.095 | 0.288 | believable |
| n=50, ?, causal_timebank_standard_eci_norule | binF1 | `qwen/qwen3-30b-a3b-thinkin`→**0.140** · `deepseek/deepseek-v3.2:nit`→0.126 · `qwen/qwen3-next-80b-a3b-th`→0.104 | 0.036 | suggestive |
| n=50, ?, causal_timebank_standard_eci_norule | binF1 | `deepseek/deepseek-v3.2:nit`→**0.105** · `openai/gpt-5.4-mini`→0.079 | 0.026 | suggestive |

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=10, ?, common_eci_extractor_tool | binF1 | `Qwen/Qwen3-Next-80B-A3B-In`→**0.399** · `deepseek/deepseek-v3.2:nit`→0.397 · `qwen/qwen3-30b-a3b-thinkin`→0.203 | 0.196 | believable |
| n=50, ?, esl_standard_eci | binF1 | `deepseek/deepseek-v3.2:nit`→**0.208** · `qwen/qwen3-30b-a3b-thinkin`→0.113 | 0.095 | believable |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, ?, maven_ere_standard_eci_intra | intraF1 | `deepseek/deepseek-v3.2`→**0.380** · `openai/gpt-4o-mini`→0.243 | 0.138 | believable |
| n=5, ?, maven_ere_standard_eci_intra | intraF1 | `google/gemini-2.5-flash-li`→**0.421** · `google/gemini-3.1-flash-li`→0.293 | 0.128 | believable |
| n=50, ?, maven_ere_standard_eci_intra | intraF1 | `openai/gpt-5.5`→**0.401** · `deepseek/deepseek-v4-pro`→0.396 · `google/gemini-2.5-flash-li`→0.340 · `google/gemini-3.1-flash-li`→0.301 | 0.100 | believable |
| n=50, ?, maven_ere_standard_eci | binF1 | `deepseek/deepseek-v3.2:nit`→**0.167** · `qwen/qwen3-30b-a3b-thinkin`→0.109 | 0.058 | suggestive |
| n=50, ?, maven_ere_standard_eci_intra | intraF1 | `deepseek/deepseek-v3.2`→**0.389** · `qwen/qwen3-30b-a3b-thinkin`→0.359 | 0.030 | suggestive |
| n=50, ?, maven_ere_standard_eci_intra | intraF1 | `z-ai/glm-5.2`→**0.404** · `deepseek/deepseek-v3.2`→0.399 | 0.005 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, ?, meci_standard_eci | binF1 | `deepseek/deepseek-v3.2:nit`→**0.571** · `qwen/qwen3-30b-a3b-thinkin`→0.333 | 0.238 | believable |
| n=50, ?, meci_standard_eci | binF1 | `deepseek/deepseek-v3.2:nit`→**0.470** · `qwen/qwen3-30b-a3b-thinkin`→0.444 · `openai/gpt-5.4-mini`→0.242 | 0.228 | believable |
| n=50, ?, meci_standard_eci | binF1 | `deepseek/deepseek-v3.2:nit`→**0.683** · `Qwen/Qwen3-Next-80B-A3B-In`→0.508 | 0.176 | believable |
| n=1, ?, meci_standard_eci | binF1 | `qwen/qwen3-30b-a3b-thinkin`→**0.533** · `qwen/qwen3.6-35b-a3b:nitro`→0.500 | 0.033 | suggestive |
| n=50, ?, meci_standard_eci | binF1 | `qwen/qwen3.6-35b-a3b:nitro`→**0.522** · `qwen/qwen3-30b-a3b-thinkin`→0.508 | 0.014 | noise |

### `dataset.max_examples`

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=?, qwen3-30b-a3b-thinking-2507:nitro, common_eci_extractor_tool | binF1 | `10`→**0.228** · `5`→0.110 | 0.117 | believable |
| n=?, deepseek-v3.2:nitro, esl_standard_eci | intraF1 | `50`→**0.245** · `5`→0.204 | 0.041 | suggestive |
| n=?, qwen3-30b-a3b-thinking-2507:nitro, esl_standard_eci | binF1 | `100`→**0.241** · `50`→0.219 | 0.022 | noise |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=?, deepseek-v3.2, common_eci_extractor_tool | intraF1 | `5`→**0.444** · `50`→0.311 | 0.134 | believable |
| n=?, deepseek-v4-pro, maven_ere_standard_eci_intra | intraF1 | `10`→**0.487** · `50`→0.384 | 0.103 | believable |
| n=?, gpt-5.5, maven_ere_standard_eci_intra | intraF1 | `10`→**0.500** · `50`→0.401 | 0.099 | believable |
| n=?, gemini-2.5-flash-lite, maven_ere_standard_eci_intra | intraF1 | `5`→**0.421** · `50`→0.340 | 0.081 | suggestive |
| n=?, gemini-3.1-flash-lite, maven_ere_standard_eci_intra | intraF1 | `50`→**0.301** · `5`→0.293 | 0.008 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=?, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `1`→**0.533** · `50`→0.345 | 0.189 | believable |
| n=?, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `716`→**0.407** · `50`→0.392 | 0.014 | noise |

### `dataset.prompt`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro | binF1 | `causal_timebank_explicit_m`→**0.442** · `causal_timebank_standard_e`→0.126 | 0.316 | believable |
| n=50, deepseek-v3.2:nitro | binF1 | `causal_timebank_explicit_m`→**0.383** · `causal_timebank_standard_e`→0.105 | 0.278 | believable |
| n=50, gpt-5.4-mini | binF1 | `causal_timebank_explicit_m`→**0.095** · `causal_timebank_standard_e`→0.079 | 0.017 | noise |

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro | binF1 | `esl_standard_eci_rewrite`→**0.249** · `esl_standard_eci`→0.185 | 0.064 | suggestive |
| n=50, deepseek-v3.2:nitro | binF1 | `esl_standard_eci_rewrite`→**0.274** · `esl_standard_eci`→0.221 | 0.053 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro | binF1 | `esl_standard_eci_norule`→**0.145** · `esl_standard_eci`→0.113 | 0.032 | suggestive |
| n=50, deepseek-v4-pro | intraF1 | `esl_standard_eci_norule_in`→**0.414** · `esl_standard_eci_intra_lea`→0.408 | 0.006 | noise |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=5, deepseek-v3.2 | intraF1 | `maven_ere_standard_eci_int`→**0.340** · `maven_ere_standard_eci_int`→0.051 | 0.290 | believable |
| n=50, deepseek-v3.2:nitro | binF1 | `maven_ere_narrative_eci`→**0.171** · `maven_ere_narrative`→0.126 | 0.045 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro | binF1 | `maven_ere_standard_eci_nor`→**0.138** · `maven_ere_standard_eci`→0.109 | 0.030 | suggestive |
| n=50, deepseek-v3.2:nitro | intraF1 | `maven_ere_standard_eci`→**0.353** · `maven_ere_standard_eci_nor`→0.336 | 0.018 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=1, qwen3.6-35b-a3b:nitro | binF1 | `meci_standard_eci_norule`→**0.727** · `meci_standard_eci`→0.500 | 0.227 | believable |
| n=50, qwen3-30b-a3b-thinking-2507:nitro | binF1 | `meci_standard_eci_steps`→**0.344** · `meci_standard_eci`→0.333 · `meci_standard_eci_norule`→0.234 | 0.110 | believable |
| n=50, deepseek-v3.2:nitro | binF1 | `meci_standard_eci`→**0.409** · `meci_standard_eci_binary`→0.346 | 0.063 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro | binF1 | `meci_standard_eci`→**0.444** · `meci_standard_eci_steps`→0.399 | 0.045 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro | binF1 | `meci_standard_eci`→**0.320** · `meci_standard_eci_binary`→0.281 | 0.039 | suggestive |
| n=50, qwen3.6-35b-a3b:nitro | binF1 | `meci_standard_eci`→**0.522** · `meci_standard_eci_norule`→0.512 | 0.010 | noise |

### `few_shot.enabled`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, causal_timebank_explicit_marker | binF1 | `True`→**0.432** · `False`→0.332 | 0.100 | believable |
| n=50, deepseek-v3.2:nitro, causal_timebank_explicit_marker | binF1 | `False`→**0.383** · `True`→0.341 | 0.042 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, causal_timebank_standard_eci_norule | binF1 | `True`→**0.140** · `False`→0.102 | 0.038 | suggestive |

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, esl_standard_eci | intraF1 | `True`→**0.362** · `False`→0.245 | 0.117 | believable |
| n=50, deepseek-v3.2:nitro, esl_standard_eci | intraF1 | `True`→**0.327** · `False`→0.220 | 0.107 | believable |
| n=50, deepseek-v3.2:nitro, esl_standard_eci | binF1 | `True`→**0.221** · `False`→0.185 | 0.036 | suggestive |
| n=10, qwen3-30b-a3b-thinking-2507:nitro, common_eci_extractor_tool | binF1 | `True`→**0.228** · `False`→0.203 | 0.025 | suggestive |
| n=50, deepseek-v3.2:nitro, esl_standard_eci_rewrite | binF1 | `True`→**0.274** · `False`→0.249 | 0.025 | suggestive |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | intraF1 | `True`→**0.356** · `False`→0.315 | 0.041 | suggestive |
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | binF1 | `True`→**0.198** · `False`→0.157 | 0.041 | suggestive |
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | intraF1 | `True`→**0.353** · `False`→0.338 | 0.016 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, Qwen3-Next-80B-A3B-Instruct, meci_standard_eci | binF1 | `False`→**0.602** · `True`→0.508 | 0.094 | believable |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.394** · `True`→0.333 | 0.061 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.444** · `True`→0.392 | 0.052 | suggestive |

### `data.novote_norel`

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, meci_standard_eci | binF1 | `True`→**0.712** · `False`→0.470 · `<unset>`→0.378 | 0.334 | believable |

### `experiment.resampling.enabled`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v4-pro, causal_timebank_explicit_marker | intraF1 | `True`→**0.475** · `False`→0.396 | 0.079 | suggestive |

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, esl_standard_eci | intraF1 | `False`→**0.362** · `True`→0.327 | 0.035 | suggestive |
| n=50, deepseek-v3.2:nitro, esl_standard_eci | intraF1 | `False`→**0.245** · `True`→0.220 | 0.025 | suggestive |
| n=50, deepseek-v4-pro, esl_standard_eci_intra_lean | intraF1 | `True`→**0.431** · `False`→0.408 | 0.023 | noise |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | intraF1 | `False`→**0.338** · `True`→0.315 | 0.023 | noise |
| n=50, deepseek-v4-pro, maven_ere_standard_eci_intra | intraF1 | `False`→**0.384** · `True`→0.375 | 0.009 | noise |
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | intraF1 | `True`→**0.356** · `False`→0.353 | 0.003 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `True`→**0.508** · `False`→0.345 | 0.164 | believable |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `True`→**0.455** · `False`→0.392 | 0.063 | suggestive |
| n=50, deepseek-v3.2:nitro, meci_standard_eci | binF1 | `True`→**0.513** · `False`→0.470 | 0.043 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `True`→**0.375** · `False`→0.333 | 0.042 | suggestive |

### `few_shot.cot_generation.enabled`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, causal_timebank_explicit_marker | binF1 | `True`→**0.442** · `False`→0.341 | 0.101 | believable |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, causal_timebank_standard_eci_norule | binF1 | `True`→**0.140** · `False`→0.078 | 0.062 | suggestive |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.455** · `True`→0.375 | 0.080 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.392** · `True`→0.333 | 0.059 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.444** · `True`→0.394 | 0.051 | suggestive |

### `few_shot.systematic`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, causal_timebank_standard_eci_norule | binF1 | `True`→**0.140** · `False`→0.105 | 0.035 | suggestive |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.431** · `True`→0.333 | 0.098 | believable |

### `data.binary_undirected`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, causal_timebank_standard_eci_norule | binF1 | `False`→**0.140** · `True`→0.071 | 0.069 | suggestive |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `False`→**0.333** · `True`→0.320 | 0.013 | noise |

### `experiment.resampling.distinct_fewshots`

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v4-pro, maven_ere_standard_eci_intra_lean | intraF1 | `True`→**0.431** · `False`→0.429 | 0.002 | noise |

### `experiment.tools`

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | binF1 | `['eci_extractor']`→**0.167** · `['coherence', 'counterfact`→0.161 | 0.005 | noise |

### `few_shot.cot_generation.num_steps`

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2, maven_ere_standard_eci_intra | intraF1 | `2`→**0.389** · `<unset>`→0.387 · `3`→0.380 | 0.009 | noise |

### `few_shot.n_examples`

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v4-pro, maven_ere_standard_eci_intra | intraF1 | `6`→**0.394** · `3`→0.384 | 0.010 | noise |

### `few_shot.pool_size`

**causal_timebank**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, causal_timebank_explicit_marker | binF1 | `20`→**0.332** · `3`→0.296 | 0.036 | suggestive |

**event_story_line**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2:nitro, esl_standard_eci | binF1 | `20`→**0.185** · `3`→0.158 | 0.027 | suggestive |
| n=50, deepseek-v3.2:nitro, esl_standard_eci_rewrite | binF1 | `20`→**0.274** · `5`→0.264 | 0.010 | noise |

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v3.2, maven_ere_standard_eci_intra | intraF1 | `3`→**0.399** · `30`→0.387 | 0.012 | noise |
| n=50, deepseek-v3.2:nitro, maven_ere_standard_eci | binF1 | `3`→**0.161** · `20`→0.157 | 0.004 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `20`→**0.450** · `3`→0.375 | 0.074 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `30`→**0.381** · `3`→0.333 | 0.048 | suggestive |
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `3`→**0.392** · `300`→0.387 | 0.005 | noise |

### `few_shot.selection`

**maven_ere**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, deepseek-v4-pro, maven_ere_standard_eci_intra | intraF1 | `similarity`→**0.396** · `bert`→0.384 | 0.012 | noise |

**meci**

| context | metric | values → score | Δ | verdict |
|---|---|---|---:|---|
| n=50, qwen3-30b-a3b-thinking-2507:nitro, meci_standard_eci | binF1 | `random`→**0.450** · `similarity`→0.385 | 0.064 | suggestive |

---

## 2-knob ablations (exactly two knobs differ)

Confounded by construction — listed so you know the data exists, not as evidence.

| knobs varied | pairs | datasets |
|---|---:|---|
| `dataset.prompt` + `model.default_model_id` | 76 | causal_timebank (36), meci (30), maven_ere (9), event_story_line (1) |
| `few_shot.cot_generation.enabled` + `few_shot.enabled` | 43 | causal_timebank (24), meci (17), maven_ere (1), event_story_line (1) |
| `few_shot.enabled` + `model.default_model_id` | 36 | meci (25), causal_timebank (7), event_story_line (2), maven_ere (2) |
| `dataset.prompt` + `few_shot.enabled` | 32 | meci (23), maven_ere (4), causal_timebank (3), event_story_line (2) |
| `few_shot.pool_size` + `model.default_model_id` | 21 | maven_ere (16), meci (5) |
| `few_shot.cot_generation.num_steps` + `model.default_model_id` | 20 | maven_ere (20) |
| `dataset.prompt` + `few_shot.cot_generation.enabled` | 18 | meci (17), causal_timebank (1) |
| `experiment.resampling.enabled` + `model.default_model_id` | 17 | meci (17) |
| `experiment.resampling.enabled` + `few_shot.enabled` | 16 | meci (8), event_story_line (4), maven_ere (4) |
| `few_shot.cot_generation.enabled` + `model.default_model_id` | 15 | meci (9), causal_timebank (6) |
| `dataset.max_examples` + `few_shot.enabled` | 14 | meci (6), maven_ere (5), event_story_line (3) |
| `few_shot.pool_size` + `few_shot.selection` | 14 | causal_timebank (10), meci (2), maven_ere (1), event_story_line (1) |
| `few_shot.cot_generation.enabled` + `few_shot.systematic` | 13 | meci (12), causal_timebank (1) |
| `dataset.max_examples` + `model.default_model_id` | 12 | maven_ere (11), meci (1) |
| `few_shot.enabled` + `few_shot.selection` | 12 | causal_timebank (10), maven_ere (1), event_story_line (1) |
| `few_shot.cot_generation.enabled` + `few_shot.pool_size` | 10 | meci (10) |
| `dataset.prompt` + `experiment.resampling.enabled` | 10 | meci (6), maven_ere (3), event_story_line (1) |
| `experiment.resampling.enabled` + `few_shot.cot_generation.enabled` | 10 | meci (10) |
| `few_shot.enabled` + `few_shot.pool_size` | 10 | meci (6), event_story_line (2), causal_timebank (1), maven_ere (1) |
| `data.binary_undirected` + `dataset.prompt` | 9 | meci (9) |
| `dataset.prompt` + `few_shot.systematic` | 9 | meci (9) |
| `dataset.max_examples` + `experiment.resampling.enabled` | 9 | meci (4), event_story_line (4), maven_ere (1) |
| `dataset.prompt` + `few_shot.pool_size` | 9 | maven_ere (4), meci (3), event_story_line (2) |
| `experiment.resampling.enabled` + `few_shot.pool_size` | 9 | meci (5), maven_ere (4) |
| `experiment.resampling.enabled` + `few_shot.systematic` | 6 | meci (6) |
| `dataset.max_examples` + `few_shot.cot_generation.enabled` | 6 | event_story_line (5), meci (1) |
| `dataset.prompt` + `experiment.tools` | 6 | event_story_line (5), maven_ere (1) |
| `few_shot.systematic` + `model.default_model_id` | 5 | meci (3), causal_timebank (2) |
| `experiment.resampling.n_runs` + `few_shot.pool_size` | 5 | meci (5) |
| `data.binary_undirected` + `few_shot.cot_generation.enabled` | 5 | meci (4), causal_timebank (1) |
| `few_shot.cot_generation.enabled` + `few_shot.selection` | 5 | event_story_line (2), causal_timebank (2), maven_ere (1) |
| `few_shot.enabled` + `few_shot.systematic` | 4 | meci (3), causal_timebank (1) |
| `data.binary_undirected` + `few_shot.systematic` | 4 | meci (3), causal_timebank (1) |
| `dataset.max_examples` + `few_shot.pool_size` | 4 | maven_ere (2), meci (1), event_story_line (1) |
| `few_shot.selection` + `model.default_model_id` | 4 | maven_ere (4) |
| `few_shot.pool_size` + `few_shot.systematic` | 3 | meci (3) |
| `data.binary_undirected` + `model.default_model_id` | 3 | causal_timebank (2), meci (1) |
| `dataset.kfold.enabled` + `dataset.kfold.n_folds` | 3 | meci (3) |
| `data.binary_undirected` + `data.vote_to_binary` | 3 | event_story_line (2), meci (1) |
| `few_shot.cot_generation.num_steps` + `few_shot.pool_size` | 3 | maven_ere (3) |
| `dataset.max_examples` + `dataset.prompt` | 3 | maven_ere (3) |
| `dataset.prompt` + `experiment.resampling.distinct_fewshots` | 3 | maven_ere (3) |
| `data.binary_undirected` + `few_shot.enabled` | 2 | causal_timebank (1), meci (1) |
| `data.binary_undirected` + `experiment.resampling.enabled` | 2 | meci (2) |
| `dataset.prompt` + `few_shot.selection` | 2 | maven_ere (2) |
| `dataset.prompt` + `experiment.enable_tools` | 2 | maven_ere (2) |

_58 distinct 2-knob combinations, 539 run pairs total._
