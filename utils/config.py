from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    base = Path(path).parent
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    active_ds = cfg["active_dataset"]
    prompt_name = cfg["datasets"][active_ds]["prompt"]
    prompt_path = base / "prompts" / f"{prompt_name}.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        cfg["prompt"] = yaml.safe_load(f)

    return cfg
