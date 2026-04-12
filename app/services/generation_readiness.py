"""Scaffold for a future OpenAI (or other) generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.config import APP_ROOT
from app.services import lecture_service
from app.services.lecture_statuses import READY_AFTER_EXTRACTION


@dataclass
class GenerationInputs:
    """Result of prepare_generation_inputs."""

    ok: bool
    reason: str
    payload: Optional[dict[str, Any]] = None


def prepare_generation_inputs(lecture_id: int) -> GenerationInputs:
    """
    Load lecture, verify extracted text exists and status allows generation.
    Does not call any model — returns a structured dict for a future pipeline.
    """
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return GenerationInputs(False, "Lecture not found.")

    status = lec["status"]
    allowed = {
        READY_AFTER_EXTRACTION,
        "generation_failed",
        "generation_complete",
        "generation_pending",  # retry if a previous run crashed mid-way
    }
    if status not in allowed:
        return GenerationInputs(
            False,
            f"Status must be {READY_AFTER_EXTRACTION!r}, generation_failed (retry), or "
            f"generation_complete (regenerate). Current: {status!r}.",
        )

    rel = lec.get("extracted_text_path")
    if not rel:
        return GenerationInputs(False, "No extracted_text_path in database.")

    path = APP_ROOT / rel
    if not path.is_file():
        return GenerationInputs(False, f"Missing file: {rel}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return GenerationInputs(False, f"Could not read extracted text: {e}")

    if not text.strip():
        return GenerationInputs(False, "Extracted text file is empty.")

    payload = {
        "lecture_id": lec["id"],
        "course_id": lec["course_id"],
        "course_name": lec["course_name"],
        "lecture_title": lec["title"],
        "source_file_name": lec["source_file_name"],
        "source_file_path": lec["source_file_path"],
        "extracted_text_path": lec["extracted_text_path"],
        "extracted_text": text,
        "status": lec["status"],
    }
    return GenerationInputs(True, "", payload)
