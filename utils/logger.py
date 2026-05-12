# FILE: utils/logger.py
from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sanitize_ts(ts: str) -> str:
    return (
        ts.replace("+00:00", "Z")
        .replace(":", "-")
        .replace("+", "p")
        .replace("/", "_")
    )

def _safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"

def _git_cmd(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git not found"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"

def capture_git_state(repo_dir: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {"is_git_repo": False}
    rc, head, err = _git_cmd(["rev-parse", "HEAD"], repo_dir)
    if rc != 0:
        state["error"] = err or "Not a git repo?"
        return state

    state["is_git_repo"] = True
    state["commit"] = head
    rc, branch, _ = _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    state["branch"] = branch if rc == 0 else "unknown"
    rc, porcelain, _ = _git_cmd(["status", "--porcelain"], repo_dir)
    state["dirty"] = bool(porcelain) if rc == 0 else None
    rc, diff, _ = _git_cmd(["diff"], repo_dir)
    state["diff"] = diff if rc == 0 else None
    return state

def make_run_stem(logdir: str, filename_prefix: str = "run") -> tuple[Path, str, str]:
    """Pre-generate the log stem so callers can set sibling file paths before the run starts.

    Returns (logs_path, stem, run_id).
    """
    logs_path = Path.cwd() / logdir
    logs_path.mkdir(parents=True, exist_ok=True)
    ts_safe = _sanitize_ts(_now_iso())
    run_id = uuid.uuid4().hex[:8]
    stem = f"{filename_prefix}_{ts_safe}_{run_id}"
    return logs_path, stem, run_id


def log_experiment(
    *,
    logdir: str = "logs",
    config: Dict[str, Any],
    cli_args: Dict[str, Any] = None,
    results: Any,
    filename_prefix: str = "run",
    _stem: Optional[str] = None,
    git_state: Optional[Dict[str, Any]] = None,
) -> str:
    root_path = Path.cwd()
    logs_path = root_path / logdir
    logs_path.mkdir(parents=True, exist_ok=True)

    if _stem:
        stem = _stem
        run_id = stem.rsplit("_", 1)[-1]
        ts_raw = _now_iso()
    else:
        ts_raw = _now_iso()
        ts_safe = _sanitize_ts(ts_raw)
        run_id = uuid.uuid4().hex[:8]
        stem = f"{filename_prefix}_{ts_safe}_{run_id}"

    # Use pre-captured git state if provided, otherwise capture now
    if git_state is None:
        git_state = capture_git_state(root_path)
    runtime_info = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(root_path),
    }

    # Construct Payload matching original schema EXACTLY
    payload = {
        "run_id": run_id,
        "timestamp_utc": ts_raw,
        "status": "success",
        "cli_args": _safe(cli_args or {}),
        "config": _safe(config),
        "results": _safe(results),
        "git": _safe(git_state),
        "runtime": runtime_info,
        "error": None,
    }

    # Write JSON
    json_path = logs_path / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write Git Patch if dirty
    if git_state["is_git_repo"] and git_state.get("diff"):
        patch_path = logs_path / f"{stem}.diff.patch"
        patch_path.write_text(git_state["diff"], encoding="utf-8")

    # Write YAML config copy
    cfg_path = logs_path / f"{stem}.config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return str(json_path)