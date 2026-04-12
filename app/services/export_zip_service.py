"""Build ZIP archives for lecture or course folders (filesystem under courses/)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.config import COURSES_DIR
from app.services import course_service, lecture_service
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.slugs import slugify


def _ensure_under_courses(path: Path) -> Path:
    """Resolve path and ensure it stays under COURSES_DIR."""
    resolved = path.resolve()
    base = COURSES_DIR.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as e:
        raise ValueError("Invalid export path") from e
    return resolved


def zip_lecture_export(lecture_id: int) -> tuple[bytes, str]:
    """
    Zip the lecture folder (source/, outputs/, meta.json, extracted_text.txt, etc.).
    Returns (raw bytes, suggested filename).
    """
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        raise FileNotFoundError("Lecture not found")
    root = _ensure_under_courses(lecture_root_from_source_relative(lec["source_file_path"]))
    if not root.is_dir():
        raise FileNotFoundError("Lecture folder not on disk")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                arc = path.relative_to(root).as_posix()
                zf.write(path, arc)

    fname = f"{lec['course_slug']}__{lec['slug']}.zip"
    return buf.getvalue(), fname


def zip_course_export(course_id: int) -> tuple[bytes, str]:
    """
    Zip the whole course directory (all lecture subfolders and course_index/, etc.).
    """
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise FileNotFoundError("Course not found")
    root = _ensure_under_courses(COURSES_DIR / course["slug"])
    if not root.is_dir():
        raise FileNotFoundError("Course folder not on disk")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                arc = path.relative_to(root).as_posix()
                zf.write(path, arc)

    fname = f"{slugify(course['name'])[:100] or course['slug']}-course-export.zip"
    return buf.getvalue(), fname
