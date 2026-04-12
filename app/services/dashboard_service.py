"""Home dashboard aggregates (stats, attention queue)."""

from __future__ import annotations

from typing import Any

from app.services import course_service, lecture_service


def get_home_dashboard() -> dict[str, Any]:
    return {
        "course_count": course_service.count_courses(),
        "lecture_count": lecture_service.count_lectures(),
        "status_counts": lecture_service.count_lectures_by_status(),
        "needs_attention": lecture_service.list_lectures_needing_attention(limit=25),
    }
