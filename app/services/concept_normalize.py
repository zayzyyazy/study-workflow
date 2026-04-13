"""Normalize concept text for deduplication and matching."""

from __future__ import annotations

import re

from app.services.concept_quality import strip_leading_numbering


def normalize_concept_key(text: str) -> str:
    """
    Canonical match key (stored in ``concepts.normalized_name``).

    Strips outline numbering first so ``1. Foo`` and ``Foo`` align; lowercases,
    collapses whitespace, trims edge punctuation.
    """
    s = strip_leading_numbering(text.strip())
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:!?-*•'\"«»()[]`")
    return s


def clean_display_name(text: str) -> str:
    """Readable label: trim, strip numbering prefixes, collapse spaces, cap length."""
    s = strip_leading_numbering(text.strip())
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t")
    if len(s) > 200:
        s = s[:197] + "…"
    return s
