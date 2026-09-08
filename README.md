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
| `syntax.level` | Append a spaCy parse summary after the document text (`off`/`mentions`/`args`/`paths`) |
| `syntax.discourse` | Whole-document sections, orthogonal to `level` (`skeleton`, `participants`) |
| `llm_cache.enabled` | Replay identical CoT-synthesis and tool calls from disk instead of the API |

Results land in `logs/allatonce/run_<timestamp>_<id>.json`.

### Response caching

Two caches make reruns cheap without letting a stale response cross experiments:

- `logs/llm_cache.sqlite` — keyed on the exact serialized message list plus the model id,
  temperature and bound tools. Covers CoT synthesis and tool sub-LLM calls.
- `logs/cot_cache.json` — keyed on doc id plus a fingerprint of the model, prompt and
  gold-formatting config. Skips a whole 2–3 call synthesis chain in one lookup.

**Inference is never cached.** The graph in `main.py` is built with `cache=False`, so reported
F1 always comes from live calls. Cached responses are stored with their token counts and cost
stripped, so they contribute nothing to `cost_usd` — the `llm_cache_hits` / `llm_cache_misses`
metrics in MLflow show how much of a run was replayed. To force everything to regenerate,
change `llm_cache.namespace` to any new string.

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

## Syntactic annotation

`syntax.level` appends a spaCy-derived annotation block directly after the document text —
in the prediction prompt, in the CoT-synthesis input and in the few-shot demonstrations
alike, so the model sees the same format everywhere. No prompt file is edited; the block is
spliced in by `utils/syntax.py`.

| Level | Contents | Typical block size |
|---|---|---|
| `off` | Nothing. Prompts are byte-identical to runs made before the feature existed. | — |
| `mentions` | Per event mention: lemma, POS, tense/verb-form/voice, dependency relation and head. | ~350 tok |
| `args` | `mentions` + each mention's subject/object/oblique arguments + per-sentence connectives. | ~700 tok |
| `paths` | `args` + the shortest dependency path between each same-sentence candidate pair. | ~1300 tok |

`syntax.discourse` is a separate, orthogonal list — set either, both or neither, with or
without a `level`:

| Component | Contents | Cost |
|---|---|---|
| `skeleton` | One row per sentence over the whole document: predicate, subject, object, polarity/modality, event count. Covers the 47% of sentences with no event mention that every `level` ignores. | O(sentences) |
| `participants` | Named entities and the sentences each recurs in — a coreference proxy for participants shared between events, and the only signal that crosses a sentence boundary. Enables the NER pipe. | O(tokens) |

Every `level` is mention-centric and `paths` is quadratic in mentions per sentence; both
discourse components are linear in the text. Running them alone (`level: off`) isolates what
whole-document context contributes.

spaCy parses the dataset's own `tokens` and `sentences`, never the `<ID surface>`-marked
`doc_text` — so mention ids map to exact tokens and sentence membership matches
`few_shot.intra_only`. `doc["doc_text"]` is never modified, keeping few-shot similarity
selection and the analysis scripts on unannotated text.

English only (`en_core_web_sm`); documents in any other language get no block, so MECI is
annotated on its 87 English documents alone — read it via `per_lang_metrics`. If spaCy is
missing the block is dropped everywhere and the run matches `off`. Blocks are capped at 8000
characters per document. The rendered block is part of the CoT cache fingerprint, so turning
the knob never replays a CoT written over unannotated text.

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
uv run mlflow server --host 0.0.0.0 --allowed-hosts jupyterhub.pagoda.liris.cnrs.fr --cors-allowed-origins https://jupyterhub.pagoda.liris.cnrs.fr --backend-store-uri sqlite:///mlflow.db --port 5000
```
```bash
uv run mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --port 5000
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
