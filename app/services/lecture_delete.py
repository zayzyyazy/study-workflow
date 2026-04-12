"""Remove lecture from DB and delete its folder on disk."""

from __future__ import annotations

import shutil

from app.config import APP_ROOT
from app.services import lecture_paths, lecture_service


def delete_lecture(lecture_id: int) -> tuple[bool, str, int | None]:
    """
    Deletes lecture row (CASCADE artifacts + lecture_concepts), then removes lecture folder.
    Returns (ok, message, course_id for redirect or None).
    """
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found.", None

    course_id = int(lec["course_id"])
    try:
        root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    except Exception:  # noqa: BLE001
        root = None

    lecture_service.delete_lecture_row(lecture_id)

    if root is not None and root.exists():
        try:
            shutil.rmtree(root)
        except OSError as e:
            return True, f"Lecture removed from the library, but folder could not be deleted: {e}", course_id

    return True, "Lecture deleted.", course_id
