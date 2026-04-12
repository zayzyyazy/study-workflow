"""Run generation for all lectures in a course that are ready_for_generation."""

from __future__ import annotations

from typing import Any

from app.services import lecture_service, openai_service
from app.services.lecture_generation import run_study_materials_generation


def run_bulk_generate_ready_in_course(course_id: int) -> dict[str, Any]:
    """
    Synchronously generate study materials for each lecture with status ready_for_generation.
    Returns counts: succeeded, failed, skipped (lectures not in ready state), ready (attempted).
    If API key missing, returns ok=False with error message.
    """
    if not openai_service.is_openai_configured():
        return {
            "ok": False,
            "error": "OpenAI is not configured. Set OPENAI_API_KEY in .env before bulk generate.",
        }

    all_lecs = lecture_service.list_lectures_for_course(course_id)
    ready = [l for l in all_lecs if l.get("status") == "ready_for_generation"]
    skipped = len(all_lecs) - len(ready)

    succeeded = 0
    failed = 0
    for lec in ready:
        ok, _msg = run_study_materials_generation(int(lec["id"]))
        if ok:
            succeeded += 1
        else:
            failed += 1

    return {
        "ok": True,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "ready": len(ready),
    }
