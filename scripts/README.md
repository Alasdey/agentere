# scripts/

Helper scripts organised by purpose. Run everything with `uv run` from the project root unless noted otherwise. All scripts resolve the project root from their own `__file__` path, so they work from any working directory.

---

## analysis/

Post-run tools that operate on the JSON log files produced by `main.py`.

### `summary.py` — aggregate runs into a spreadsheet

Scans the logs tree recursively and writes one row per run to an Excel file.

```bash
uv run scripts/analysis/summary.py                         # defaults: logs/ → runs_summary.xlsx
uv run scripts/analysis/summary.py --logs logs/allatonce --out out.xlsx
```

Columns include model, dataset, split, max_examples, tools, resampling, micro/macro F1, binary metrics, per-label metrics, skipped docs.

---

### `recompute_metrics.py` — recompute metrics in-place on log files

Recomputes all metric fields directly from `per_pair_predictions`, skipping `gold="unannotated"` pairs. Useful when the metric logic has changed since a run was written, or when you need to add a missing metric. Rewrites `per_label`, `macro_f1`, `micro_*`, `binary`, `total_pairs`, `per_doc_metrics`, and `per_lang_metrics` inside `results{}`. `per_pair_predictions` itself is left unchanged.

```bash
uv run scripts/analysis/recompute_metrics.py                         # all MECI logs in logs/
uv run scripts/analysis/recompute_metrics.py path/to/run.json ...   # specific files
uv run scripts/analysis/recompute_metrics.py --dry-run               # print diff, no writes
```

---

### `fix_unannotated.py` — mark unannotated pairs in existing logs

Patches `gold` fields that should be `"unannotated"` in older log files that predate that label. Run before `recompute_metrics.py` on those files.

```bash
uv run scripts/analysis/fix_unannotated.py logs/allatonce/run_XYZ.json
```

---

### `binary_eval.py` — direction-agnostic binary evaluation

Loads a log file and computes undirected binary metrics (any relation = Causal, NoRel = NoRel). Supports two aggregation modes:

- **OR (lenient)**: a pair is Causal if either direction is Causal.
- **AND (conservative)**: both directions must agree (single-direction pairs count as-is).

```bash
uv run scripts/analysis/binary_eval.py logs/allatonce/run_XYZ.json
uv run scripts/analysis/binary_eval.py logs/allatonce/run_XYZ.json --mode and
```

---

### `show_traces.py` — pretty-print sampled LLM traces

Reads `.traces.sample.jsonl` files produced during runs and renders each message in a trace (system, human, AI, tool calls/results) with colour-coded headers and token usage info. Useful for inspecting what was actually sent to and received from the model.

```bash
uv run scripts/analysis/show_traces.py                              # latest sample file in logs/
uv run scripts/analysis/show_traces.py path/to/run.traces.sample.jsonl
uv run scripts/analysis/show_traces.py --all                        # every sample file found
uv run scripts/analysis/show_traces.py --trace 2                    # only trace #2 (1-indexed)
uv run scripts/analysis/show_traces.py --width 120                  # wrap width
uv run scripts/analysis/show_traces.py --full                       # don't truncate content
uv run scripts/analysis/show_traces.py --no-color
uv run scripts/analysis/show_traces.py --logs logs/allatonce        # custom logs root
```

---

### `posteval.py` — quick per-file undirected binary report

Prints a concise per-language and overall binary summary for one or more log files. Lighter than `binary_eval.py`, good for a quick sanity check.

```bash
uv run scripts/analysis/posteval.py logs/allatonce/run_XYZ.json
```

---

## experiments/

Standalone experiment runners. These are self-contained and do not share state with the main pipeline.

### `llm_perpair.py` — pairwise inference variant

Runs the LLM on one pair at a time instead of the full document at once. Slower but isolates each pair from cross-pair contamination. Useful for ablation.

### `enc_samp_perc.py` / `enc_perpair_samp_perc.py` — encoder learning curves

Train the Longformer classifier at varying fractions of the training set and record F1 at each checkpoint. Produces a `results_curve.jsonl` log. Results land in `scripts/experiments/logs/`.

### `autoprompt.py` — automatic prompt optimisation

Iteratively rewrites the system/user prompt using a meta-LLM critic. Evaluates each candidate on a sample of documents and keeps the best-performing variants. Configure via the `OPTIM_CONFIG` dict at the top of the file.

---

## dev/

Sanity checks and interactive utilities. Not intended for production use.

| Script | Purpose |
|---|---|
| `dataprep_test.py` | Loads a dataset split and prints a few parsed documents to verify annotation parsing |
| `display_dataset.py` | Prints raw HuggingFace rows for any dataset/split |
| `model_test.py` | Fires a single LLM call through the LangGraph graph to verify model connectivity and tool routing |
| `openrouter.py` | Checks the current OpenRouter API key status and quota |
| `all_use.py` | Runs a small end-to-end inference loop with LangSmith tracing enabled; useful for tracing setup validation |
