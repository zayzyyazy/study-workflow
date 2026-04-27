"""User-authored summaries (Zusammenfassungen) under each course: ``courses/<slug>/summaries/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import COURSES_DIR

SUMMARIES_DIRNAME = "summaries"
_ALLOWED_EXT = frozenset({".md", ".txt", ".markdown", ".pdf"})


def summaries_root(course_slug: str) -> Path:
    return COURSES_DIR / course_slug / SUMMARIES_DIRNAME


def ensure_summaries_dir(course_slug: str) -> Path:
    p = summaries_root(course_slug)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_summary_basename(name: str) -> str | None:
    raw = (name or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw:
        return None
    base = Path(raw).name
    if not base or base.startswith("."):
        return None
    suf = Path(base).suffix.lower()
    if suf not in _ALLOWED_EXT:
        return None
    return base


def list_summary_files(course_slug: str) -> list[dict[str, Any]]:
    root = summaries_root(course_slug)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _ALLOWED_EXT:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            {
                "name": p.name,
                "size_bytes": int(st.st_size),
                "modified": st.st_mtime,
            }
        )
    return out


def resolved_summary_path(course_slug: str, basename: str) -> Path | None:
    b = safe_summary_basename(basename)
    if not b:
        return None
    root = summaries_root(course_slug).resolve()
    path = (summaries_root(course_slug) / b).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path
