from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from utils.logger import capture_git_state

# Cost per 1 million tokens (input, output) in USD.
# OpenRouter prices — update as needed.
_PRICING: Dict[str, tuple[float, float]] = {
    "openai/chatgpt-4o-latest":                    (5.00,  15.00),
    "openai/gpt-5-mini":                           (0.40,   1.60),
    "deepseek/deepseek-r1-0528":                   (0.55,   2.19),
    "deepseek/deepseek-v3.2":                      (0.27,   1.10),
    "google/gemini-2.5-flash-lite-preview-09-2025":(0.10,   0.40),
    "qwen/qwen3-30b-a3b-instruct-2507":            (0.10,   0.30),
    "mistralai/mistral-small-2603":                (0.10,   0.30),
    "mistralai/mistral-small-3.2-24b-instruct":    (0.10,   0.30),
    "mistralai/ministral-3b-2512":                 (0.04,   0.10),
    "x-ai/grok-4.1-fast":                         (2.00,  10.00),
    "moonshotai/kimi-k2.5":                        (0.14,   0.55),
    "nvidia/nemotron-3-super-120b-a12b":           (0.40,   0.40),
    "nvidia/nemotron-3-nano-30b-a3b":              (0.09,   0.18),
    "allenai/olmo-3.1-32b-instruct":               (0.10,   0.20),
    "allenai/olmo-3.1-32b-think":                  (0.10,   0.20),
}


def _parse_token_usage(trace_path: Path) -> Dict[str, int]:
    """Sum input/output/cache_read tokens across all AI messages in the trace file."""
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    with gzip.open(trace_path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            messages = json.loads(raw)
            for msg in messages:
                if msg.get("id", [""])[-1] != "AIMessage":
                    continue
                usage = msg.get("kwargs", {}).get("usage_metadata")
                if not usage:
                    continue
                totals["input_tokens"]       += usage.get("input_tokens", 0)
                totals["output_tokens"]      += usage.get("output_tokens", 0)
                totals["cache_read_tokens"]  += usage.get("input_token_details", {}).get("cache_read", 0)
    return totals


def log_run(
    *,
    config: Dict[str, Any],
    final_report: Dict[str, Any],
    outfile: str,
    trace_path: Optional[Path],
    run_name: str,
) -> str:
    """Log a completed run to MLflow. Returns the MLflow run ID."""
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "agentere"))

    ds_key = config["active_dataset"]
    ds_cfg = config["datasets"][ds_key]
    fs_cfg = config.get("few_shot", {})
    exp_cfg = config["experiment"]
    model_id = config["model"]["default_model_id"]

    with mlflow.start_run(run_name=run_name) as run:
        # ── Params ──────────────────────────────────────────────────────────
        mlflow.log_params({
            "model":               model_id,
            "dataset":             ds_cfg["name"],
            "dataset_split":       ds_cfg.get("split", "test"),
            "max_examples":        ds_cfg.get("max_examples", 0),
            "temperature":         config["model"]["temperature"],
            "prompt":              ds_cfg.get("prompt", ""),
            "few_shot_enabled":    fs_cfg.get("enabled", False),
            "few_shot_n":          fs_cfg.get("n_examples", 0),
            "few_shot_selection":  fs_cfg.get("selection", "random"),
            "resampling_enabled":  exp_cfg["resampling"]["enabled"],
            "resampling_n_runs":   exp_cfg["resampling"]["n_runs"],
            "enable_tools":        exp_cfg.get("enable_tools", False),
            "tools":               str(exp_cfg.get("tools", [])),
            "concurrency":         exp_cfg.get("concurrency", 1),
            "retries":             exp_cfg.get("retries", 0),
        })

        # ── Performance metrics ──────────────────────────────────────────────
        metrics: Dict[str, float] = {
            "micro_f1":        final_report["micro_f1"],
            "macro_f1":        final_report["macro_f1"],
            "micro_precision": final_report["micro_precision"],
            "micro_recall":    final_report["micro_recall"],
            "total_pairs":     final_report["total_pairs"],
            "skipped_docs":    final_report["skipped_docs"],
        }
        bin_m = final_report.get("binary", {})
        metrics["binary_f1"]        = bin_m.get("f1", 0.0)
        metrics["binary_precision"] = bin_m.get("precision", 0.0)
        metrics["binary_recall"]    = bin_m.get("recall", 0.0)
        for label, lm in final_report.get("per_label", {}).items():
            safe = label.lower().replace(" ", "_")
            metrics[f"{safe}_f1"]        = lm.get("f1", 0.0)
            metrics[f"{safe}_precision"] = lm.get("precision", 0.0)
            metrics[f"{safe}_recall"]    = lm.get("recall", 0.0)

        # ── Token usage + cost ───────────────────────────────────────────────
        if trace_path and trace_path.exists():
            usage = _parse_token_usage(trace_path)
            metrics["tokens_input"]       = usage["input_tokens"]
            metrics["tokens_output"]      = usage["output_tokens"]
            metrics["tokens_cache_read"]  = usage["cache_read_tokens"]

            price = _PRICING.get(model_id)
            if price:
                price_in, price_out = price
                # cache_read is billed at ~10% of input price on most providers
                cost = (
                    (usage["input_tokens"] - usage["cache_read_tokens"]) * price_in / 1_000_000
                    + usage["cache_read_tokens"] * price_in * 0.1 / 1_000_000
                    + usage["output_tokens"] * price_out / 1_000_000
                )
                metrics["cost_usd"] = round(cost, 6)

        mlflow.log_metrics(metrics)

        # ── Git tags ────────────────────────────────────────────────────────
        git = capture_git_state(Path.cwd())
        if git.get("is_git_repo"):
            mlflow.set_tags({
                "git.commit": git.get("commit", ""),
                "git.branch": git.get("branch", ""),
                "git.dirty":  str(git.get("dirty", False)),
            })

        # ── Artifacts ───────────────────────────────────────────────────────
        mlflow.log_artifact(outfile)
        if trace_path and trace_path.exists():
            mlflow.log_artifact(str(trace_path))

        return run.info.run_id
