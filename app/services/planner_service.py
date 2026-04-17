"""Deterministic study planner — dates, schedule, lecture progress (no AI)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Optional

from app.services import lecture_service, planner_schedule_service, topic_deep_dive

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Full labels for schedule editor (Monday = 0 … Sunday = 6, matches date.weekday())
WEEKDAY_NAMES_FORM = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def _parse_hhmm(s: str) -> time:
    parts = (s or "00:00").strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return time(h, m)


def _dt(d: date, t: time) -> datetime:
    return datetime.combine(d, t)


def _occurs_on_day(item: dict[str, Any], day: date) -> bool:
    if item.get("recurrence") == "once":
        sd = (item.get("specific_date") or "").strip()
        return bool(sd) and sd == day.isoformat()
    if item.get("recurrence") == "weekly":
        wd = item.get("weekday")
        if wd is None:
            return False
        return int(wd) == day.weekday()
    return False


def _expand_instances(
    schedule: list[dict[str, Any]],
    start_day: date,
    num_days: int,
    *,
    after: Optional[datetime] = None,
) -> list[tuple[datetime, dict[str, Any]]]:
    """(start datetime, raw row) for each occurrence in range."""
    out: list[tuple[datetime, dict[str, Any]]] = []
    after = after or datetime.min
    for offset in range(num_days):
        day = start_day + timedelta(days=offset)
        for s in schedule:
            if not _occurs_on_day(s, day):
                continue
            st = _parse_hhmm(str(s["start_time"]))
            dt = _dt(day, st)
            if dt >= after.replace(second=0, microsecond=0):
                out.append((dt, s))
    out.sort(key=lambda x: x[0])
    return out


def _format_slot(row: dict[str, Any], day: date) -> str:
    wd = WEEKDAY_NAMES[day.weekday()]
    return f"{wd} {day.isoformat()} · {row['start_time']}–{row['end_time']} · {row['title']}"


def build_planner_dashboard(now: Optional[datetime] = None) -> dict[str, Any]:
    now = now or datetime.now()
    today = now.date()
    tnow = now.time()

    schedule = planner_schedule_service.list_schedule_items()
    lectures = lecture_service.list_lectures_for_planner()

    by_course: dict[int, list[dict[str, Any]]] = {}
    for lec in lectures:
        by_course.setdefault(int(lec["course_id"]), []).append(lec)

    # Today's schedule rows
    today_rows: list[dict[str, Any]] = []
    for s in schedule:
        if not _occurs_on_day(s, today):
            continue
        today_rows.append(dict(s))
    today_rows.sort(key=lambda r: str(r["start_time"]))

    # Now
    now_lines: list[dict[str, Any]] = []
    in_block: Optional[dict[str, Any]] = None
    for row in today_rows:
        st = _parse_hhmm(str(row["start_time"]))
        et = _parse_hhmm(str(row["end_time"]))
        if st <= tnow <= et:
            in_block = row
            break
    if in_block:
        k = str(in_block.get("kind") or "")
        now_lines.append(
            {
                "text": f"In session: {in_block['title']}",
                "href": _course_href(in_block),
                "sub": k,
            }
        )
    else:
        now_lines.append({"text": "No scheduled block right now.", "href": None, "sub": None})

    next_today: Optional[dict[str, Any]] = None
    for row in today_rows:
        st = _parse_hhmm(str(row["start_time"]))
        if _dt(today, st) > now:
            next_today = row
            break
    if next_today:
        now_lines.append(
            {
                "text": f"Next today: {next_today['title']} · {next_today['start_time']}",
                "href": _course_href(next_today),
                "sub": str(next_today.get("kind") or ""),
            }
        )

    # Today — study lines (short)
    today_study: list[dict[str, Any]] = []
    seen_href: set[str] = set()

    def _add_line(text: str, href: str | None, sub: str | None) -> None:
        if href and href in seen_href:
            return
        if href:
            seen_href.add(href)
        today_study.append({"text": text, "href": href, "sub": sub})

    course_ids_today: set[int] = set()
    for row in today_rows:
        if str(row.get("kind")) == "lecture" and row.get("course_id"):
            course_ids_today.add(int(row["course_id"]))

    for cid in sorted(course_ids_today):
        for lec in by_course.get(cid, []):
            sp = lec.get("study_progress") or "not_started"
            if sp == "not_started":
                _add_line(
                    f"Prepare: {lec['title']}",
                    f"/lectures/{lec['id']}",
                    str(lec.get("course_name") or ""),
                )
            elif sp == "in_progress":
                _add_line(
                    f"Continue: {lec['title']}",
                    f"/lectures/{lec['id']}",
                    str(lec.get("course_name") or ""),
                )

    for row in today_rows:
        if str(row.get("kind")) == "project":
            _add_line(
                f"Project block: {row['title']}",
                _course_href(row),
                "scheduled",
            )

    for lec in lectures:
        if int(lec.get("is_starred") or 0) and (lec.get("study_progress") or "") != "done":
            _add_line(
                f"Starred: {lec['title']}",
                f"/lectures/{lec['id']}",
                str(lec.get("course_name") or ""),
            )

    # Next up (7 days)
    upcoming = _expand_instances(schedule, today, 8, after=now)
    next_up: list[dict[str, Any]] = []
    for dt, row in upcoming[:14]:
        day = dt.date()
        next_up.append(
            {
                "when": dt.strftime("%a %Y-%m-%d %H:%M"),
                "label": row["title"],
                "kind": row.get("kind"),
                "href": _course_href(row),
            }
        )

    # Catch-up
    catch_up: list[dict[str, Any]] = []
    for lec in lectures:
        if lec.get("study_progress") == "in_progress":
            catch_up.append(
                {
                    "text": f"Finish: {lec['title']}",
                    "href": f"/lectures/{lec['id']}",
                    "sub": str(lec.get("course_name") or ""),
                }
            )

    # Deadlines (one-off, date >= today)
    deadlines: list[dict[str, Any]] = []
    for s in schedule:
        if str(s.get("kind")) != "deadline":
            continue
        if str(s.get("recurrence")) != "once":
            continue
        sd = (s.get("specific_date") or "").strip()
        if not sd:
            continue
        try:
            d = date.fromisoformat(sd)
        except ValueError:
            continue
        if d >= today:
            deadlines.append(
                {
                    "text": f"{s['title']} · {sd}",
                    "href": _course_href(s),
                    "sub": f"{s.get('start_time')}",
                    "_sort": d,
                }
            )
    deadlines.sort(key=lambda x: x["_sort"])
    for d in deadlines:
        d.pop("_sort", None)

    # Deep dives missing
    deep_rows = topic_deep_dive.list_missing_recommended_deep_dives(10)
    deep_dive_lines: list[dict[str, Any]] = []
    for d in deep_rows:
        lid = int(d["lecture_id"])
        slug = d["slug"]
        deep_dive_lines.append(
            {
                "text": f"Deep dive: {d['topic_title']} ({d['lecture_title']})",
                "href": f"/lectures/{lid}/topics/{slug}",
                "sub": str(d.get("course_name") or ""),
            }
        )

    # One-line stats
    n_ip = sum(1 for lec in lectures if lec.get("study_progress") == "in_progress")
    n_ns = sum(1 for lec in lectures if lec.get("study_progress") == "not_started")
    n_done = sum(1 for lec in lectures if lec.get("study_progress") == "done")

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "weekday_label": WEEKDAY_NAMES[today.weekday()],
        "today_iso": today.isoformat(),
        "now_lines": now_lines,
        "today_schedule": [
            {
                "text": f"{r['start_time']}–{r['end_time']} · {r['title']}",
                "href": _course_href(r),
                "sub": f"{r.get('kind') or ''}"
                + (f" · {r.get('course_name')}" if r.get("course_name") else ""),
            }
            for r in today_rows
        ],
        "today_study": today_study,
        "next_up": next_up,
        "catch_up": catch_up[:16],
        "deadlines": deadlines,
        "deep_dive_lines": deep_dive_lines,
        "stats_line": f"In progress: {n_ip} · Not started: {n_ns} · Done: {n_done}",
        "schedule_items": schedule,
        "today_rows_raw": today_rows,
    }


def _course_href(row: dict[str, Any]) -> str | None:
    cid = row.get("course_id")
    if cid is None:
        return None
    try:
        return f"/courses/{int(cid)}"
    except (TypeError, ValueError):
        return None
