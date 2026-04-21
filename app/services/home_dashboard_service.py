"""Home page: compact, high-signal study snapshot (deterministic)."""

from __future__ import annotations

from typing import Any

from app.services import (
    course_service,
    lecture_links_service,
    lecture_service,
    planner_service,
    topic_deep_dive,
    uni_task_service,
)


def build_home_dashboard() -> dict[str, Any]:
    lectures = lecture_service.list_lectures_for_planner()
    dash = planner_service.build_planner_dashboard()
    uni_tasks_open = uni_task_service.list_tasks(status="open", limit=20)
    task_lecture_ids = {int(t["lecture_id"]) for t in uni_tasks_open if isinstance(t.get("lecture_id"), int)}
    task_course_ids = {int(t["course_id"]) for t in uni_tasks_open if isinstance(t.get("course_id"), int)}

    continue_ = [l for l in lectures if l.get("study_progress") == "in_progress"]
    continue_.sort(
        key=lambda x: (
            1 if int(x.get("id") or 0) in task_lecture_ids else 0,
            1 if int(x.get("course_id") or 0) in task_course_ids else 0,
            -(int(x.get("is_starred") or 0)),
            x.get("created_at") or "",
        ),
        reverse=True,
    )
    continue_ = continue_[:6]

    not_started = [l for l in lectures if l.get("study_progress") == "not_started"]
    not_started.sort(
        key=lambda x: (
            1 if int(x.get("id") or 0) in task_lecture_ids else 0,
            1 if int(x.get("course_id") or 0) in task_course_ids else 0,
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
    next_actions = dash.get("next_actions") or []
    next_actions = next_actions[:5]

    connection_hints = lecture_links_service.home_connection_hints(limit=5)
    uni_tasks_open = uni_tasks_open[:8]
    uni_tasks_done = uni_task_service.list_tasks(status="done", limit=4)

    return {
        "continue_lectures": continue_,
        "not_started_pick": not_started,
        "recent_lectures": recent,
        "courses_attention": attention,
        "deep_dive_picks": deep_picks,
        "deep_dive_by_course": deep_by_course,
        "planner_next": planner_next,
        "planner_focus": focus,
        "next_actions": next_actions,
        "connection_hints": connection_hints,
        "uni_tasks_open": uni_tasks_open,
        "uni_tasks_done": uni_tasks_done,
        "stats_line": dash.get("stats_line", ""),
    }
