# agentere

LLM-based Event Causality Identification (ECI) pipeline. Given a document and a set of event mention pairs, the model predicts the directed causal relation between each pair (`CauseEffect`, `EffectCause`, `CAUSE`, `PRECONDITION`, `FALLING_ACTION`, or `NoRel`) depending on the active dataset.

The pipeline runs via OpenRouter, supports optional tool augmentation (encoder predictions, few-shot examples, coherence rules, counterfactual checks), and writes structured JSON logs for every run.

---

## Setup

```bash
# Install dependencies
uv sync

# Set credentials (required)
read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY
```

---

## Running

```bash
uv run main.py
```

Edit `config.yaml` to control what runs. The key knobs:

| Key | What it does |
|---|---|
| `model.default_model_id` | OpenRouter model string |
| `active_dataset` | Which dataset to evaluate (`meci`, `maven_ere`, `event_story_line`) |
| `datasets.<key>.max_examples` | How many documents to run (0 = all) |
| `experiment.enable_tools` | Whether the LLM can call tools |
| `experiment.tools` | Which tools to expose |
| `experiment.resampling.enabled` | Run N passes per doc and take majority vote |
| `few_shot.enabled` | Inject training examples before the LLM call |
| `few_shot.selection` | `random` or `similarity` (TF-IDF cosine) |

Results land in `logs/allatonce/run_<timestamp>_<id>.json`.

---

## Datasets

All datasets are loaded from HuggingFace (`Nofing/*`). Annotations are parsed from an inline relation-triple format: `<SRC_ID text> LABEL <TGT_ID text>`.

| Key | Dataset | Labels |
|---|---|---|
| `meci` | MECI-v0.1 (multilingual, EN/DA/ES/TR/UR) | CauseEffect, EffectCause |
| `maven_ere` | MAVEN-ERE Causal Events | CAUSE, PRECONDITION |
| `event_story_line` | EventStoryLine 1.5 | PRECONDITION, FALLING_ACTION |

`NoRel` is always present as the negative class and excluded from macro/micro metrics.

---

## Tools

Tools are LangGraph nodes the LLM can call during inference. They are defined in `tools/` and registered in `tools/__init__.py`.

| Tool | Description |
|---|---|
| `few_shot_examples` | Retrieves labelled training examples (random or TF-IDF similarity) |
| `encoder` | Provides predictions from a pre-trained Longformer classifier with per-class confidence scores |
| `coherence` | Checks the predicted relation graph against dataset-specific transitivity/symmetry rules |
| `counterfactual_check` | Runs a but-for counterfactual test on a specific pair via a second LLM call |
| `bare_causes` | Reduces the document to its bare causal skeleton via a second LLM call |
| `eci` | Identifies all mentions causally related to a target (linear-complexity alternative to pairwise) |

Few-shot can also be injected **systematically** before the LLM call (bypassing the tool mechanism) by setting `few_shot.systematic: true`.

---

## Log format

Each run produces up to three files sharing the same stem (`run_<ts>_<id>`):

| File | Contents |
|---|---|
| `.json` | Full payload: config snapshot, git state, global metrics, per-doc metrics, `per_pair_predictions`, per-language metrics |
| `.traces.jsonl.gz` | Gzipped JSONL — one line per LLM call, full message sequence |
| `.traces.sample.jsonl` | First 5 traces uncompressed, for quick inspection |
| `.diff.patch` | Git diff at run time (only written if the working tree is dirty) |
| `.config.yaml` | Copy of `config.yaml` at run time |

The primary evaluation field is `results.per_pair_predictions`: a list of `{doc_idx, id, lang, pair, gold, pred, vote_counts}` rows, one per (document, mention-pair).

To visualize the traces:
```bash
uv run mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts jupyterhub.pagoda.liris.cnrs.fr --cors-allowed-origins https://jupyterhub.pagoda.liris.cnrs.fr
```

---

## Project layout

```
main.py                  # Entrypoint — async pipeline, concurrency, logging
config.yaml              # All runtime configuration
prompts/                 # Prompt YAML files (system + user template per dataset/variant)
dataprep/dataprep.py     # HuggingFace dataset loading and annotation parsing
model/model.py           # LangGraph chat graph (LLM + optional tool loop)
tools/                   # LangChain tools (few_shot, encoder, coherence, …)
utils/
  config.py              # Config loader (merges config.yaml + prompt file)
  formatting.py          # pair_lines and gold output formatting
  metrics.py             # Multiclass and binary metric computation
  reporting.py           # Per-doc/per-lang aggregation → run report dict
  resample.py            # Majority-vote aggregation across N runs
  logger.py              # JSON/YAML/patch log writer
  trace_dump.py          # Per-call gzipped trace writer
encoder_baseline/        # Standalone Longformer classifier (training + inference)
scripts/                 # Post-run analysis, experiments, dev utilities (see scripts/README.md)
derelict/                # Dead code kept for reference
logs/                    # Run outputs (gitignored)
```

---

## Encoder baseline

`encoder_baseline/encoder.py` is a standalone PyTorch training script for a Longformer-based pair classifier. It produces a `predictions.json` file that the `encoder` tool reads at inference time. Set `encoder.path` in `config.yaml` to point to the right predictions file.
