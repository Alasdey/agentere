#!/usr/bin/env python3
"""
Migrate MLflow runs + traces from file store (mlruns/) to a SQLite backend.

Bypasses the slow MLflow Python API for traces: reads YAML/flat files directly
from disk and INSERTs into SQLite in bulk. Runs use the API (only ~57 runs).

Span data (traces.json) stays on disk — each trace has an absolute
mlflow.artifactLocation tag so the server finds it without moving anything.

Usage (from repo root):
    uv run python scripts/migrate_mlflow_to_sqlite.py

Then test:
    uv run mlflow server --backend-store-uri sqlite:///mlflow_test.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 --port 5000 \
        --allowed-hosts jupyterhub.pagoda.liris.cnrs.fr \
        --cors-allowed-origins https://jupyterhub.pagoda.liris.cnrs.fr

If the UI is fast, swap mlflow_test.db → mlflow.db and update config.yaml.
"""

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

SRC_URI = "mlruns"
DST_URI = "sqlite:///mlflow_test.db"
EXPERIMENT_NAME = "agentere"


def migrate_runs(src: MlflowClient, dst: MlflowClient, src_exp_id: str, dst_exp_id: str):
    runs = src.search_runs(experiment_ids=[src_exp_id], max_results=10_000)
    print(f"Migrating {len(runs)} runs …")
    for i, run in enumerate(runs, 1):
        info = run.info
        new_run = dst.create_run(
            experiment_id=dst_exp_id,
            start_time=info.start_time,
            run_name=info.run_name,
            tags=run.data.tags,
        )
        rid = new_run.info.run_id
        for k, v in run.data.params.items():
            dst.log_param(rid, k, v)
        for k, v in run.data.metrics.items():
            dst.log_metric(rid, k, v)
        dst.set_terminated(rid, status=info.status, end_time=info.end_time)
        print(f"  [{i:3d}/{len(runs)}] {info.run_name}")


def _read_flat_dir(path: Path) -> dict:
    if not path.exists():
        return {}
    return {f.name: f.read_text(encoding="utf-8").strip() for f in path.iterdir() if f.is_file()}


def migrate_traces_direct(src_exp_id: str, dst_exp_integer_id: int, db_path: str):
    traces_dir = Path(SRC_URI) / src_exp_id / "traces"
    if not traces_dir.exists():
        print("  No traces directory found, skipping.")
        return

    trace_dirs = [d for d in traces_dir.iterdir() if d.is_dir()]
    total = len(trace_dirs)
    print(f"  Found {total} trace directories, parsing …")

    info_rows, tag_rows, meta_rows = [], [], []
    t0 = time.time()

    for i, trace_dir in enumerate(trace_dirs, 1):
        info_path = trace_dir / "trace_info.yaml"
        if not info_path.exists():
            continue

        info = yaml.safe_load(info_path.read_text())
        trace_id = info["trace_id"]

        rt = info.get("request_time", "")
        try:
            ts_ms = int(datetime.fromisoformat(rt.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            ts_ms = 0

        exec_ms = info.get("execution_duration_ms")
        status = info.get("state", "OK")
        req_preview = (info.get("request_preview") or "")[:1000] or None
        resp_preview = (info.get("response_preview") or "")[:1000] or None

        info_rows.append((trace_id, dst_exp_integer_id, ts_ms, exec_ms, status, None, req_preview, resp_preview))

        for k, v in _read_flat_dir(trace_dir / "tags").items():
            tag_rows.append((k, v[:8000], trace_id))
        for k, v in _read_flat_dir(trace_dir / "request_metadata").items():
            meta_rows.append((k, v[:8000], trace_id))

        if i % 500 == 0:
            elapsed = time.time() - t0
            eta = (total - i) / (i / elapsed)
            print(f"  [{i}/{total}] parsed … ETA {eta:.0f}s")

    print(f"  Parsed {total} traces in {time.time() - t0:.1f}s, inserting into SQLite …")
    t1 = time.time()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO trace_info "
            "(request_id, experiment_id, timestamp_ms, execution_time_ms, status, "
            "client_request_id, request_preview, response_preview) "
            "VALUES (?,?,?,?,?,?,?,?)",
            info_rows,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO trace_tags (key, value, request_id) VALUES (?,?,?)",
            tag_rows,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO trace_request_metadata (key, value, request_id) VALUES (?,?,?)",
            meta_rows,
        )
    conn.close()
    print(f"  Inserted {len(info_rows)} traces in {time.time() - t1:.1f}s.")


def migrate():
    src = MlflowClient(tracking_uri=SRC_URI)

    src_exp = src.get_experiment_by_name(EXPERIMENT_NAME)
    if src_exp is None:
        print(f"ERROR: experiment '{EXPERIMENT_NAME}' not found in {SRC_URI}", file=sys.stderr)
        sys.exit(1)
    print(f"Source: {SRC_URI}  experiment_id={src_exp.experiment_id}")

    db_path = DST_URI.replace("sqlite:///", "")
    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"Removed existing {db_path}")

    # Artifact root points at existing mlruns experiment dir so the server can
    # resolve new run artifacts. Trace span artifacts resolve via the per-trace
    # mlflow.artifactLocation tag (absolute path) that was written at trace time.
    src_artifact_root = str(Path(SRC_URI).resolve() / src_exp.experiment_id)
    mlflow.set_tracking_uri(DST_URI)
    dst_exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=src_artifact_root)
    print(f"Created SQLite experiment  experiment_id={dst_exp_id}")

    dst = MlflowClient(tracking_uri=DST_URI)

    migrate_runs(src, dst, src_exp.experiment_id, dst_exp_id)
    print()
    print("Migrating traces directly from filesystem …")
    migrate_traces_direct(src_exp.experiment_id, int(dst_exp_id), db_path)

    print(f"\nDone. Test with:")
    print(f"  uv run mlflow server --backend-store-uri {DST_URI} "
          f"--default-artifact-root ./mlruns "
          f"--host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    migrate()
