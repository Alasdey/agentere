"""
Document-level context variables for the pipeline.

Set once per document in process_document_resampled(); propagate automatically
to every asyncio child task (tool calls, few-shot generation, etc.) for that
document without threading them through every function signature.

Use doc_context() as an async context manager — it guarantees all variables
are reset even if an exception is raised inside the with-block.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Dict, FrozenSet

# ── Per-document context variables ───────────────────────────────────────────

CURRENT_DOC_ID: ContextVar[str] = ContextVar("current_doc_id", default="")
CURRENT_DOC_TEXT: ContextVar[str] = ContextVar("current_doc_text", default="")
CURRENT_DOC_MENTIONS: ContextVar[FrozenSet[str]] = ContextVar("current_doc_mentions", default=frozenset())
# mention_id -> sentence index (from gold tokenization); read-only, never mutated.
CURRENT_DOC_MENTION_SENTENCE: ContextVar[Dict[str, int]] = ContextVar("current_doc_mention_sentence", default={})
CURRENT_DOC_FOLD: ContextVar[int] = ContextVar("current_doc_fold", default=-1)
CURRENT_USER_PROMPT: ContextVar[str] = ContextVar("current_user_prompt", default="")


@asynccontextmanager
async def doc_context(
    doc_id: str,
    doc_text: str,
    mentions: FrozenSet[str],
    user_prompt: str,
    fold: int,
    mention_sentence: Dict[str, int] | None = None,
):
    """Set all document context variables for the duration of one document's inference.

    Guarantees reset on both normal exit and exception — the try/finally is
    required by the contextmanager protocol: without it, an exception raised
    inside `async with doc_context(...)` would propagate out through the
    generator's throw() and skip every line after yield.
    """
    t_id      = CURRENT_DOC_ID.set(doc_id)
    t_text    = CURRENT_DOC_TEXT.set(doc_text)
    t_ment    = CURRENT_DOC_MENTIONS.set(mentions)
    t_ms      = CURRENT_DOC_MENTION_SENTENCE.set(mention_sentence or {})
    t_prompt  = CURRENT_USER_PROMPT.set(user_prompt)
    t_fold    = CURRENT_DOC_FOLD.set(fold)
    try:
        yield
    finally:
        CURRENT_DOC_ID.reset(t_id)
        CURRENT_DOC_TEXT.reset(t_text)
        CURRENT_DOC_MENTIONS.reset(t_ment)
        CURRENT_DOC_MENTION_SENTENCE.reset(t_ms)
        CURRENT_USER_PROMPT.reset(t_prompt)
        CURRENT_DOC_FOLD.reset(t_fold)
