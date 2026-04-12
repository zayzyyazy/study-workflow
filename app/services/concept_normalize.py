"""Normalize concept text for deduplication and matching."""

from __future__ import annotations

import re


def normalize_concept_key(text: str) -> str:
    """
    Lowercase, trim, collapse whitespace, strip simple punctuation edges.
    Used as the canonical match key (stored in concepts.normalized_name).
    """
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:!?-*•'\"«»()[]`")
    return s


def clean_display_name(text: str) -> str:
    """Light cleanup for display; keeps original casing mostly."""
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 200:
        s = s[:197] + "…"
    return s
