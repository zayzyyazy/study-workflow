"""Readable slug helpers for URLs and filesystem paths."""

import re
from typing import Optional


def slugify(text: str, max_length: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    if not text:
        text = "item"
    return text[:max_length]


def sanitize_folder_name(name: str, max_length: int = 100) -> str:
    """Human-readable folder name: strip unsafe chars, collapse spaces."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "Untitled"
    return name[:max_length]
