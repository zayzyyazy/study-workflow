"""CRUD for planner schedule items (SQLite)."""

from __future__ import annotations

from typing import Any, Optional

from app.db.database import get_connection

VALID_KINDS = frozenset({"lecture", "project", "block", "deadline"})
VALID_RECURRENCE = frozenset({"weekly", "once"})


def list_schedule_items() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT s.id, s.course_id, s.title, s.kind, s.recurrence, s.weekday, s.specific_date,
                   s.start_time, s.end_time, s.created_at,
                   c.name AS course_name
            FROM planner_schedule_items s
            LEFT JOIN courses c ON c.id = s.course_id
            ORDER BY (s.weekday IS NULL), s.weekday, (s.specific_date IS NULL), s.specific_date, s.start_time, s.id
            """
        )
        return [dict(row) for row in cur.fetchall()]


def add_schedule_item(
    *,
    title: str,
    kind: str,
    recurrence: str,
    start_time: str,
    end_time: str,
    course_id: Optional[int] = None,
    weekday: Optional[int] = None,
    specific_date: Optional[str] = None,
) -> tuple[bool, str]:
    title = (title or "").strip()
    if not title:
        return False, "Title is required."
    if kind not in VALID_KINDS:
        return False, "Invalid kind."
    if recurrence not in VALID_RECURRENCE:
        return False, "Invalid recurrence."
    if recurrence == "weekly":
        if weekday is None or not (0 <= int(weekday) <= 6):
            return False, "Pick a weekday for a weekly block."
        specific_date = None
    else:
        weekday = None
        sd = (specific_date or "").strip()
        if not sd:
            return False, "Pick a date for a one-off block."
        specific_date = sd

    st = _normalize_hhmm(start_time)
    et = _normalize_hhmm(end_time)
    if not st or not et:
        return False, "Start and end time must look like HH:MM."
    if st >= et:
        return False, "End time must be after start time."

    cid: Optional[int] = None
    if course_id is not None:
        try:
            cid = int(course_id)
        except (TypeError, ValueError):
            cid = None
        if cid is not None and cid <= 0:
            cid = None

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO planner_schedule_items
            (course_id, title, kind, recurrence, weekday, specific_date, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (cid, title, kind, recurrence, weekday, specific_date, st, et),
        )
        conn.commit()
    return True, "Schedule block added."


def delete_schedule_item(item_id: int) -> tuple[bool, str]:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM planner_schedule_items WHERE id = ?", (item_id,))
        conn.commit()
        if cur.rowcount == 0:
            return False, "Entry not found."
    return True, "Removed."


def _normalize_hhmm(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    parts = s.replace(".", ":").split(":")
    if len(parts) < 2:
        return ""
    try:
        h = int(parts[0])
        m = int(parts[1])
    except ValueError:
        return ""
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return ""
    return f"{h:02d}:{m:02d}"
