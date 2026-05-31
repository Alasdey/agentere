from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into base, returning a new dict."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_prompt(cfg: Dict, base: Path) -> Dict:
    active_ds = cfg["active_dataset"]
    prompt_name = cfg["datasets"][active_ds]["prompt"]
    prompt_path = base / "prompts" / f"{prompt_name}.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        cfg["prompt"] = yaml.safe_load(f)
    return cfg


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    base = Path(path).parent
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_prompt(cfg, base)


def load_config_with_overrides(overrides: Dict, path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml, deep-merge overrides on top, then resolve the active prompt."""
    base = Path(path).parent
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deep_merge(cfg, overrides)
    return _resolve_prompt(cfg, base)
