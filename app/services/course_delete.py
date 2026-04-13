"""Remove a course and all its lectures from DB and disk."""

from __future__ import annotations

import shutil

from app.config import COURSES_DIR
from app.db.database import get_connection
from app.services import course_service, lecture_paths, lecture_service


def delete_course(course_id: int) -> tuple[bool, str]:
    """
    Fully delete a course:
      1. Delete each lecture's folder from disk (best-effort).
      2. Delete the course row — CASCADE removes lectures, artifacts, lecture_concepts.
      3. Delete the course folder on disk (holds course_index/ etc.).

    Returns (ok, message).
    """
    course = course_service.get_course_by_id(course_id)
    if not course:
        return False, "Course not found."

    course_name = course["name"]

    # 1. Delete individual lecture folders from disk
    lectures = lecture_service.list_lectures_for_course(course_id)
    for lec in lectures:
        try:
            root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
            if root.exists():
                shutil.rmtree(root)
        except Exception:  # noqa: BLE001
            pass  # best effort — DB row still gets removed below

    # 2. Delete course row (CASCADE cleans up lectures, artifacts, lecture_concepts)
    with get_connection() as conn:
        conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()

    # 3. Delete course folder (may still contain course_index/ or other leftovers)
    course_folder = COURSES_DIR / course["slug"]
    if course_folder.exists():
        try:
            shutil.rmtree(course_folder)
        except OSError as e:
            return (
                True,
                f'Course "{course_name}" removed from library, but folder could not be deleted: {e}',
            )

    return True, f'Course "{course_name}" and all its lectures have been deleted.'
