"""Save uploads under courses/ with the agreed folder layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO

from app.config import COURSES_DIR
from app.services.slugs import sanitize_folder_name


def build_lecture_directory_name(lecture_index: int, lecture_title: str) -> str:
    """
    Human-readable, filesystem-safe folder name under the course directory.

    Format ``Lecture NN - {title}`` keeps lectures ordered and stable; the DB
    stores the display title separately from the URL slug.
    """
    title_part = sanitize_folder_name(lecture_title, max_length=80)
    return f"Lecture {lecture_index:02d} - {title_part}"


def ensure_lecture_paths(
    course_folder_name: str,
    lecture_folder_name: str,
) -> tuple[Path, Path, Path]:
    """
    Returns (lecture_root, source_dir, outputs_dir).
    """
    base = COURSES_DIR / course_folder_name / lecture_folder_name
    source_dir = base / "source"
    outputs_dir = base / "outputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return base, source_dir, outputs_dir


def save_uploaded_file(
    file_obj: BinaryIO,
    dest_path: Path,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as out:
        shutil.copyfileobj(file_obj, out)


def write_meta_json(lecture_root: Path, payload: dict[str, Any]) -> Path:
    meta_path = lecture_root / "meta.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def write_extracted_text(lecture_root: Path, text: str) -> Path:
    path = lecture_root / "extracted_text.txt"
    path.write_text(text, encoding="utf-8")
    return path
