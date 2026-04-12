"""Resolve filesystem paths for a lecture from stored relative paths."""

from pathlib import Path

from app.config import APP_ROOT


def lecture_root_from_source_relative(source_file_path: str) -> Path:
    """
    source_file_path is relative to project root, e.g. courses/foo/Lecture 01/source/file.pdf.
    Lecture root is the parent of `source/`.
    """
    p = (APP_ROOT / source_file_path).resolve()
    return p.parent.parent


def source_dir_from_root(lecture_root: Path) -> Path:
    return lecture_root / "source"
