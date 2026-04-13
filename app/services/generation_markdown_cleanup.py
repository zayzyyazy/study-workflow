"""Lightweight post-processing for model-generated Markdown before saving to disk."""

from __future__ import annotations

import re


def cleanup_generated_markdown(text: str) -> str:
    """
    Safe, conservative cleanup only — no semantic rewriting.

    - Normalizes newlines
    - Strips trailing whitespace on each line
    - Collapses runs of more than two blank lines
    """
    if not text:
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
