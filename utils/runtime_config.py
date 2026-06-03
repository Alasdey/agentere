"""
Central runtime config registry.

Usage:
  # In main (or run_queue via main):
  from utils.runtime_config import set_cfg
  set_cfg(config)  # call once per run, before any tool is used

  # In tools:
  from utils.runtime_config import get_cfg, register_reset
  register_reset(my_cache_clear_fn)   # called at module level
  cfg = get_cfg()                     # called inside functions

set_cfg() stores the merged config and fires all registered reset callbacks,
so tool caches (lru_cache, lazy loads) are invalidated between queue runs.

get_cfg() lazy-loads config.yaml from disk if set_cfg() has not been called
yet (e.g. direct `python main.py` or scripts that bypass run_queue).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

_cfg: dict = {}
_reset_callbacks: list[Callable] = []


def set_cfg(cfg: dict) -> None:
    global _cfg
    _cfg = cfg
    for cb in _reset_callbacks:
        cb()


def get_cfg() -> dict:
    global _cfg
    if not _cfg:
        import yaml
        path = Path(__file__).parent.parent / "config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def register_reset(callback: Callable) -> None:
    _reset_callbacks.append(callback)
