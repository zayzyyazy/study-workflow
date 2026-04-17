"""Home page: compact, high-signal study snapshot (deterministic)."""

from __future__ import annotations

from typing import Any

from app.services import course_service, lecture_service, planner_service, topic_deep_dive


def build_home_dashboard() -> dict[str, Any]:
    lectures = lecture_service.list_lectures_for_planner()
    dash = planner_service.build_planner_dashboard()

    continue_ = [l for l in lectures if l.get("study_progress") == "in_progress"]
    continue_.sort(
        key=lambda x: (
            -(int(x.get("is_starred") or 0)),
            x.get("created_at") or "",
        ),
        reverse=True,
    )
    continue_ = continue_[:6]

    not_started = [l for l in lectures if l.get("study_progress") == "not_started"]
    not_started.sort(
        key=lambda x: (
            -(int(x.get("is_starred") or 0)),
            x.get("created_at") or "",
        ),
        reverse=True,
    )
    not_started = not_started[:6]

    recent = sorted(
        lectures,
        key=lambda x: x.get("created_at") or "",
        reverse=True,
    )[:5]

    courses = course_service.list_courses_for_home_dashboard()
    attention: list[dict[str, Any]] = []
    for c in courses:
        lc = int(c.get("lecture_count") or 0)
        done = int(c.get("study_done_count") or 0)
        if lc <= 0:
            continue
        left = lc - done
        if left <= 0:
            continue
        attention.append(
            {
                "name": c["name"],
                "href": f"/courses/{c['id']}",
                "note": f"{left} not done",
            }
        )
    attention = attention[:6]

    deep_picks = topic_deep_dive.list_missing_recommended_deep_dives(5)
    deep_by_course = topic_deep_dive.missing_deep_dives_by_course_summary()[:4]

    planner_next = dash.get("next_up") or []
    planner_next = planner_next[:4]
    focus = dash.get("focus_lines") or []
    focus = focus[:4]

    return {
        "continue_lectures": continue_,
        "not_started_pick": not_started,
        "recent_lectures": recent,
        "courses_attention": attention,
        "deep_dive_picks": deep_picks,
        "deep_dive_by_course": deep_by_course,
        "planner_next": planner_next,
        "planner_focus": focus,
        "stats_line": dash.get("stats_line", ""),
    }
