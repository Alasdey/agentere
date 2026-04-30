from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from utils.logger import capture_git_state


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

    with mlflow.start_run(run_name=run_name) as run:
        # ── Params ──────────────────────────────────────────────────────────
        mlflow.log_params({
            "model":               config["model"]["default_model_id"],
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

        # ── Metrics ─────────────────────────────────────────────────────────
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
