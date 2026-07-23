"""
Prompt-keyed LLM response cache (SQLite).

A cache entry is keyed on the exact bytes sent to the provider — the serialized message
list — plus LangChain's `llm_string`, which carries the model id, temperature and bound
tools. Two calls therefore share an entry only if they would have produced the same API
request against the same model, so a cached response can never leak across models,
prompts or tool sets the way the doc-keyed CoT cache in tools/few_shot.py used to.

Scope is deliberate: the cache is attached per LLM instance (`ChatOpenAI(cache=...)`),
not globally via set_llm_cache(). The CoT-synthesis graph and the tool sub-LLMs get it;
the main inference graph is built with `cache=False` so headline metrics are always
produced by live API calls.

Billing metadata is stripped from every entry before it is stored, so replayed calls
report zero tokens and zero cost. utils/mlflow_tracker._parse_token_usage sums usage
straight out of the trace files — without stripping, a cached run would claim tokens it
never paid for.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Optional, Sequence

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.outputs import Generation

from utils.runtime_config import get_cfg, register_reset

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         TEXT PRIMARY KEY,
    model       TEXT,
    created_at  TEXT,
    generations TEXT
)
"""


class SqliteLLMCache(BaseCache):
    """BaseCache over a single SQLite file.

    The async methods are inherited from BaseCache, which runs the sync ones in an
    executor — right for SQLite, which has no async driver here.
    """

    def __init__(self, path: str, namespace: str = "") -> None:
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: call_model is a sync graph node, so lookups arrive on
        # LangGraph's worker threads. WAL + the lock below keep concurrent writes safe at
        # experiment.concurrency (50).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def _key(self, prompt: str, llm_string: str) -> str:
        return sha256("\x00".join((self.namespace, llm_string, prompt)).encode("utf-8")).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        key = self._key(prompt, llm_string)
        with self._lock:
            row = self._conn.execute(
                "SELECT generations FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            self.misses += 1
            return None

        try:
            generations = [loads(blob) for blob in json.loads(row[0])]
        except Exception as e:
            # Deserialization failures (a LangChain upgrade changing the message schema,
            # a truncated write) must be loud but must not abort a run: drop the entry and
            # treat it as a miss so the call goes to the provider.
            print(f"[llm_cache] Dropping undeserializable entry {key[:12]}: {e}")
            with self._lock:
                self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                self._conn.commit()
            self.misses += 1
            return None

        self.hits += 1
        return generations

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        stored = [_strip_billing_metadata(gen) for gen in return_val]
        blob = json.dumps([dumps(gen) for gen in stored])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, created_at, generations) VALUES (?, ?, ?, ?)",
                (self._key(prompt, llm_string), _model_of(return_val), datetime.now().isoformat(timespec="seconds"), blob),
            )
            self._conn.commit()

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = f"{100 * self.hits / total:.0f}%" if total else "n/a"
        return f"{self.hits} hits / {self.misses} misses ({rate})"


def _strip_billing_metadata(gen: Generation) -> Generation:
    """Copy `gen` with token usage and provider cost removed, tagged as a replay.

    Deep-copied rather than mutated in place: the same object is on its way back to the
    caller and into the trace file, where the *live* call's real token counts must survive.
    """
    stored = gen.model_copy(deep=True)
    message = getattr(stored, "message", None)
    if message is None:  # plain (non-chat) Generation — no usage to strip
        return stored
    message.usage_metadata = None
    message.response_metadata.pop("token_usage", None)
    message.response_metadata["cache_hit"] = True
    return stored


def _model_of(generations: Sequence[Generation]) -> str:
    """Serving model for the `model` column — inspection and pruning only, never part of
    the key. Provider-reported, so absent on some backends."""
    for gen in generations:
        message = getattr(gen, "message", None)
        if message is not None:
            return message.response_metadata.get("model_name", "")
    return ""


@lru_cache(maxsize=1)
def get_llm_cache() -> Optional[SqliteLLMCache]:
    """Process-wide cache instance, or None when disabled.

    None is what ChatOpenAI(cache=None) already means ("use the global cache", which is
    never set), so callers can pass the result through unconditionally.
    """
    cfg = get_cfg()["llm_cache"]
    if not cfg["enabled"]:
        return None
    return SqliteLLMCache(path=cfg["path"], namespace=cfg["namespace"])


register_reset(get_llm_cache.cache_clear)
