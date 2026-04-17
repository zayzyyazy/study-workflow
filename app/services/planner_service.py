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

    # Tomorrow: same logic for prep (compact)
    tomorrow = today + timedelta(days=1)
    tomorrow_rows: list[dict[str, Any]] = []
    for s in schedule:
        if not _occurs_on_day(s, tomorrow):
            continue
        tomorrow_rows.append(dict(s))
    tomorrow_rows.sort(key=lambda r: str(r["start_time"]))

    course_ids_tomorrow: set[int] = set()
    for row in tomorrow_rows:
        if str(row.get("kind")) == "lecture" and row.get("course_id"):
            course_ids_tomorrow.add(int(row["course_id"]))

    tomorrow_study: list[dict[str, Any]] = []
    seen_t: set[str] = set()

    def _add_t(text: str, href: str | None, sub: str | None) -> None:
        if href and href in seen_t:
            return
        if href:
            seen_t.add(href)
        tomorrow_study.append({"text": text, "href": href, "sub": sub})

    for cid in sorted(course_ids_tomorrow):
        for lec in by_course.get(cid, []):
            sp = lec.get("study_progress") or "not_started"
            if sp == "done":
                continue
            if sp == "not_started":
                _add_t(
                    f"Tonight / tomorrow AM: skim «{lec['title']}»",
                    f"/lectures/{lec['id']}",
                    str(lec.get("course_name") or ""),
                )
            else:
                _add_t(
                    f"Tomorrow: finish «{lec['title']}»",
                    f"/lectures/{lec['id']}",
                    str(lec.get("course_name") or ""),
                )

    # Next up (7 days)
    upcoming = _expand_instances(schedule, today, 8, after=now)
    next_up: list[dict[str, Any]] = []
    for dt, row in upcoming[:14]:
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

    # Deep dives missing (detail list — capped)
    deep_rows = topic_deep_dive.list_missing_recommended_deep_dives(24)
    deep_dive_lines: list[dict[str, Any]] = []
    for d in deep_rows[:8]:
        lid = int(d["lecture_id"])
        slug = d["slug"]
        deep_dive_lines.append(
            {
                "text": f"{d['topic_title']} · {d['lecture_title']}",
                "href": f"/lectures/{lid}/topics/{slug}",
                "sub": str(d.get("course_name") or ""),
            }
        )
    deep_dive_by_course = topic_deep_dive.missing_deep_dives_by_course_summary()[:6]

    # Focus: next class + strongest library ties (max 4 short lines)
    focus_lines: list[dict[str, Any]] = []
    upcoming_inst = _expand_instances(schedule, today, 10, after=now)
    for dt, row in upcoming_inst:
        if str(row.get("kind")) != "lecture" or not row.get("course_id"):
            continue
        if dt > now + timedelta(hours=96):
            break
        cid = int(row["course_id"])
        cname = str(row.get("course_name") or "").strip() or "Course"
        lecs_nd = [
            l
            for l in lectures
            if int(l["course_id"]) == cid and (l.get("study_progress") or "") != "done"
        ]
        lecs_nd.sort(
            key=lambda x: (
                0 if x.get("study_progress") == "in_progress" else 1,
                -(int(x.get("is_starred") or 0)),
                x.get("title") or "",
            )
        )
        if lecs_nd:
            focus_lines.append(
                {
                    "text": f"Next «{cname}» class {dt.strftime('%a %H:%M')}: work on «{lecs_nd[0]['title']}»",
                    "href": f"/lectures/{lecs_nd[0]['id']}",
                    "sub": "your upload",
                }
            )
        for d in deep_rows:
            if int(d["course_id"]) == cid:
                focus_lines.append(
                    {
                        "text": f"Generate deep dive: {d['topic_title']}",
                        "href": f"/lectures/{d['lecture_id']}/topics/{d['slug']}",
                        "sub": d.get("lecture_title"),
                    }
                )
                break
        break

    if not focus_lines and deep_rows:
        d0 = deep_rows[0]
        focus_lines.append(
            {
                "text": f"Recommended deep dive: {d0['topic_title']}",
                "href": f"/lectures/{d0['lecture_id']}/topics/{d0['slug']}",
                "sub": str(d0.get("course_name") or ""),
            }
        )

    focus_lines = focus_lines[:4]

    # Courses that need attention (unfinished work + missing dives)
    course_attention: list[dict[str, Any]] = []
    for c in deep_dive_by_course:
        course_attention.append(
            {
                "text": f"{c['course_name']}: {c['count']} recommended deep dive(s) missing",
                "href": f"/courses/{c['course_id']}",
                "sub": "from your topic maps",
            }
        )
    for cid, lecs in by_course.items():
        total = len(lecs)
        if total < 2:
            continue
        undone = sum(1 for l in lecs if (l.get("study_progress") or "") != "done")
        if undone == 0 or undone == total:
            continue
        name = str(lecs[0].get("course_name") or "")
        if any(int(x["course_id"]) == cid for x in deep_dive_by_course):
            continue
        course_attention.append(
            {
                "text": f"{name}: {undone}/{total} lectures not done",
                "href": f"/courses/{cid}",
                "sub": "mixed progress",
            }
        )
        if len(course_attention) >= 6:
            break

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
        "today_study": today_study[:10],
        "tomorrow_schedule": [
            {
                "text": f"{r['start_time']}–{r['end_time']} · {r['title']}",
                "href": _course_href(r),
                "sub": f"{r.get('kind') or ''}"
                + (f" · {r.get('course_name')}" if r.get("course_name") else ""),
            }
            for r in tomorrow_rows
        ],
        "tomorrow_study": tomorrow_study[:8],
        "focus_lines": focus_lines,
        "course_attention": course_attention[:8],
        "deep_dive_by_course": deep_dive_by_course,
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
