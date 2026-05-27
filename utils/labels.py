"""
Canonical label constants for the pipeline.

All HF datasets use exactly these three strings in their annotation fields.
Nothing else in the codebase should hardcode "norel", "causes", etc. as
string literals — import from here instead.
"""
from __future__ import annotations

# ── The three uniform relation labels stored in every HF dataset ─────────────
CAUSES    = "causes"
CAUSED_BY = "causedby"
NOREL     = "norel"

# ── Label produced after binary collapse (causes + causedby → one positive) ──
BINARY_POS = "causal"

# ── All strings that mean "no relation", matched case-insensitively via .lower()
# Covers model outputs that deviate slightly from the canonical "norel".
NOREL_VARIANTS: frozenset = frozenset({"norel", "none", "no_rel", "unknown"})

# ── Full directed label set (non-binary mode) ─────────────────────────────────
DIRECTED_LABELS: list = [CAUSES, CAUSED_BY, NOREL]

# ── Full binary label set (binary_undirected mode) ────────────────────────────
BINARY_LABELS: list = [BINARY_POS, NOREL]
