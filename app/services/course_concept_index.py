"""Wire extraction → DB → course file after generation (never raises to callers)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.services import course_service, lecture_service
from app.services import concept_extraction, concept_service, course_index_service
from app.services.lecture_paths import lecture_root_from_source_relative

log = logging.getLogger(__name__)


def index_lecture_after_generation(lecture_id: int) -> Tuple[bool, Optional[str]]:
    """
    Extract concepts from outputs/, replace lecture_concepts, refresh course concept_index.md.
    Returns (True, None) on success, (False, message) on recoverable failure.
    """
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found for indexing."

    try:
        root = lecture_root_from_source_relative(lec["source_file_path"])
    except Exception as e:  # noqa: BLE001
        return False, f"Bad lecture path: {e}"

    outputs_dir = root / "outputs"
    names = concept_extraction.extract_concepts_from_outputs(outputs_dir)
    # Empty list still clears old lecture_concepts if materials were removed
    ok, err = concept_service.replace_lecture_concepts(lecture_id, names)
    if not ok:
        return False, err or "Could not save concepts."

    course_id = int(lec["course_id"])
    course = course_service.get_course_by_id(course_id)
    if course:
        course_index_service.write_course_concept_index_file(
            course["slug"],
            course["name"],
            course_id,
        )
    return True, None


def index_lecture_safe(lecture_id: int) -> str:
    """
    For use after generation: returns empty string on success, or a short user-facing warning.
    """
    try:
        ok, err = index_lecture_after_generation(lecture_id)
        if ok:
            return ""
        log.warning("Concept indexing failed for lecture %s: %s", lecture_id, err)
        return err or "Concept indexing failed."
    except Exception as e:  # noqa: BLE001
        log.exception("Concept indexing crashed for lecture %s", lecture_id)
        return str(e)
