"""Deterministic study planner — dates, schedule, lecture progress (no AI)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Optional

from app.services import lecture_service, planner_schedule_service, topic_deep_dive, uni_task_service

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


def _planner_kind_display(kind: str | None) -> str:
    k = (kind or "").strip().lower()
    if k == "lecture":
        return "Lecture"
    if k == "ubung":
        return "Übung"
    if k == "project":
        return "Project"
    if k in ("block", "deadline"):
        return "Project"
    return (kind or "—").strip() or "—"


def _is_course_session(kind: str | None) -> bool:
    """Lecture or Übung: link to a course and should drive prep / class-adjacent hints."""
    k = (kind or "").strip().lower()
    return k in ("lecture", "ubung")


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


# Weekly learning grid: 08:00–22:00 local time (minutes from midnight)
_WGRID_LO = 8 * 60
_WGRID_HI = 22 * 60
_WGRID_LEN = _WGRID_HI - _WGRID_LO


def _minutes_from_midnight(t: time) -> int:
    return t.hour * 60 + t.minute


def _hhmm_from_minutes_abs(m: int) -> str:
    m = max(0, min(24 * 60 - 1, m))
    return f"{m // 60:02d}:{m % 60:02d}"


def _week_dates_monday(today: date) -> list[date]:
    monday = today - timedelta(days=today.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


def _clip_to_learning_grid(sm: int, em: int) -> tuple[int, int] | None:
    a = max(sm, _WGRID_LO)
    b = min(em, _WGRID_HI)
    if b <= a:
        return None
    return (a, b)


def _merge_busy_union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping [start,end) minute ranges (absolute from midnight)."""
    iv = [(a, b) for a, b in intervals if b > a]
    if not iv:
        return []
    iv.sort(key=lambda x: x[0])
    out: list[list[int]] = [[iv[0][0], iv[0][1]]]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(x[0], x[1]) for x in out]


def _free_segments_in_grid(merged_busy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Gaps inside [_WGRID_LO, _WGRID_HI) not covered by merged busy intervals."""
    cursor = _WGRID_LO
    free: list[tuple[int, int]] = []
    for a, b in merged_busy:
        aa = max(a, _WGRID_LO)
        bb = min(b, _WGRID_HI)
        if bb <= aa:
            continue
        if cursor < aa:
            free.append((cursor, aa))
        cursor = max(cursor, bb)
    if cursor < _WGRID_HI:
        free.append((cursor, _WGRID_HI))
    return free


def build_weekly_learning_grid(
    today: date,
    schedule: list[dict[str, Any]],
    lectures: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Mon–Sun columns, 08:00–22:00: blocked vs free time from planner_schedule_items,
    plus catch-up lecture candidates (not done).
    """
    week_days = _week_dates_monday(today)
    mon, sun = week_days[0], week_days[-1]
    week_label = f"{mon.strftime('%b %d')} – {sun.strftime('%b %d, %Y')}"

    days_out: list[dict[str, Any]] = []
    for day in week_days:
        raw_busy: list[tuple[int, int, dict[str, Any]]] = []
        union_parts: list[tuple[int, int]] = []
        for s in schedule:
            if not _occurs_on_day(s, day):
                continue
            st = _parse_hhmm(str(s["start_time"]))
            et = _parse_hhmm(str(s["end_time"]))
            sm = _minutes_from_midnight(st)
            em = _minutes_from_midnight(et)
            clipped = _clip_to_learning_grid(sm, em)
            if not clipped:
                continue
            a, b = clipped
            raw_busy.append((sm, em, s))
            union_parts.append((a, b))

        merged = _merge_busy_union(union_parts)
        free = _free_segments_in_grid(merged)

        busy_blocks: list[dict[str, Any]] = []
        for sm, em, row in raw_busy:
            cl = _clip_to_learning_grid(sm, em)
            if not cl:
                continue
            a, b = cl
            top_pct = (a - _WGRID_LO) / _WGRID_LEN * 100.0
            h_pct = (b - a) / _WGRID_LEN * 100.0
            kind = str(row.get("kind") or "")
            busy_blocks.append(
                {
                    "top_pct": round(top_pct, 3),
                    "height_pct": round(max(h_pct, 0.35), 3),  # min height so short slots stay visible
                    "start": _hhmm_from_minutes_abs(a),
                    "end": _hhmm_from_minutes_abs(b),
                    "title": str(row.get("title") or "—"),
                    "kind": kind,
                    "kind_label": _planner_kind_display(kind),
                    "course_name": str(row.get("course_name") or "").strip() or None,
                }
            )

        free_lines: list[dict[str, str]] = []
        for a, b in free:
            dur_min = b - a
            if dur_min < 10:
                continue
            h, m = divmod(dur_min, 60)
            if h and m:
                extra = f"{h}h {m}m"
            elif h:
                extra = f"{h}h"
            else:
                extra = f"{m}m"
            free_lines.append(
                {
                    "start": _hhmm_from_minutes_abs(a),
                    "end": _hhmm_from_minutes_abs(b),
                    "label": f"{_hhmm_from_minutes_abs(a)}–{_hhmm_from_minutes_abs(b)} ({extra})",
                }
            )

        days_out.append(
            {
                "date_iso": day.isoformat(),
                "weekday": WEEKDAY_NAMES[day.weekday()],
                "weekday_long": WEEKDAY_NAMES_FORM[day.weekday()],
                "is_today": day == today,
                "busy_blocks": busy_blocks,
                "free_segments": free_lines,
            }
        )

    catch_rows: list[dict[str, Any]] = []
    for lec in lectures:
        sp = lec.get("study_progress") or "not_started"
        if sp == "done":
            continue
        mk = str(lec.get("material_kind") or "lecture")
        catch_rows.append(
            {
                "id": int(lec["id"]),
                "title": str(lec.get("title") or ""),
                "course_name": str(lec.get("course_name") or ""),
                "progress": sp,
                "material_kind": mk,
                "href": f"/lectures/{int(lec['id'])}",
            }
        )
    catch_rows.sort(
        key=lambda x: (
            0 if x["progress"] == "in_progress" else 1,
            0 if x["material_kind"] == "lecture" else 1,
            (x["course_name"] or "").lower(),
            x["title"].lower(),
        )
    )

    return {
        "week_label": week_label,
        "grid_start_label": "08:00",
        "grid_end_label": "22:00",
        "days": days_out,
        "catch_up_lectures": catch_rows[:24],
    }


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
        now_lines.append(
            {
                "text": f"In session: {in_block['title']}",
                "href": _course_href(in_block),
                "sub": _planner_kind_display(str(in_block.get("kind") or "")),
            }
        )
    else:
        now_lines.append({"text": "No scheduled session right now.", "href": None, "sub": None})

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
                "sub": _planner_kind_display(str(next_today.get("kind") or "")),
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
        if _is_course_session(str(row.get("kind"))) and row.get("course_id"):
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
                f"Project: {row['title']}",
                _course_href(row),
                "scheduled",
            )
        elif str(row.get("kind")) == "ubung" and row.get("course_id"):
            _add_line(
                f"Übung today: {row['title']}",
                _course_href(row),
                str(row.get("course_name") or ""),
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
        if _is_course_session(str(row.get("kind"))) and row.get("course_id"):
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
                "kind_label": _planner_kind_display(str(row.get("kind") or "")),
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

    # Open uni tasks (used across prioritization)
    open_tasks = uni_task_service.list_tasks(status="open", limit=64)
    open_tasks_by_course: dict[int, list[dict[str, Any]]] = {}
    open_tasks_by_lecture: dict[int, list[dict[str, Any]]] = {}
    for t in open_tasks:
        cid = t.get("course_id")
        lid = t.get("lecture_id")
        if isinstance(cid, int):
            open_tasks_by_course.setdefault(cid, []).append(t)
        if isinstance(lid, int):
            open_tasks_by_lecture.setdefault(lid, []).append(t)

    # Deep dives missing (detail list — capped) — skips done lectures by default
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
        if not _is_course_session(str(row.get("kind"))) or not row.get("course_id"):
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
            linked_tasks = open_tasks_by_lecture.get(int(lecs_nd[0]["id"]), [])
            if linked_tasks:
                focus_lines.append(
                    {
                        "text": f"Task first: {linked_tasks[0]['title']}",
                        "href": f"/lectures/{lecs_nd[0]['id']}",
                        "sub": "linked uni task",
                    }
                )
            else:
                c_tasks = open_tasks_by_course.get(cid, [])
                if c_tasks:
                    focus_lines.append(
                        {
                            "text": f"Course task: {c_tasks[0]['title']}",
                            "href": f"/courses/{cid}",
                            "sub": "open uni task",
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

    # Top next actions: merges timing, lecture state, and uni tasks into one short ranked list
    next_actions_scored: list[tuple[float, dict[str, Any]]] = []

    # Task urgency first
    for t in open_tasks:
        score = 22.0
        due_label = str(t.get("due_label") or "")
        if due_label.startswith("Overdue"):
            score += 20.0
        elif due_label == "Due today":
            score += 16.0
        elif due_label == "Due tomorrow":
            score += 12.0
        if t.get("lecture_id"):
            score += 8.0
        elif t.get("course_id"):
            score += 5.0
        if str(t.get("task_kind") or "") in {"exercise", "review", "prepare"}:
            score += 3.0
        href: str | None = None
        if t.get("lecture_id"):
            href = f"/lectures/{int(t['lecture_id'])}"
        elif t.get("course_id"):
            href = f"/courses/{int(t['course_id'])}"
        sub = "uni task"
        if t.get("due_label"):
            sub = f"{sub} · {t['due_label']}"
        if t.get("course_name"):
            sub = f"{sub} · {t['course_name']}"
        next_actions_scored.append(
            (
                score,
                {
                    "text": t["title"],
                    "href": href,
                    "sub": sub,
                    "kind": "task",
                },
            )
        )

    # Pre-class prep for tomorrow / near-term classes
    for dt, row in upcoming_inst[:16]:
        if not _is_course_session(str(row.get("kind"))) or not row.get("course_id"):
            continue
        cid = int(row["course_id"])
        soon_hours = (dt - now).total_seconds() / 3600.0
        if soon_hours > 36:
            continue
        lecs_nd = [
            l
            for l in lectures
            if int(l["course_id"]) == cid and (l.get("study_progress") or "") != "done"
        ]
        if not lecs_nd:
            continue
        lecs_nd.sort(
            key=lambda x: (
                0 if x.get("study_progress") == "in_progress" else 1,
                -(int(x.get("is_starred") or 0)),
                x.get("title") or "",
            )
        )
        lead = lecs_nd[0]
        score = 18.0 + max(0.0, 14.0 - (soon_hours / 3.0))
        if lead.get("study_progress") == "in_progress":
            score += 4.0
        next_actions_scored.append(
            (
                score,
                {
                    "text": f"Before class: review «{lead['title']}»",
                    "href": f"/lectures/{lead['id']}",
                    "sub": f"{dt.strftime('%a %H:%M')} · {lead.get('course_name') or ''}",
                    "kind": "before_class",
                },
            )
        )
        c_tasks = open_tasks_by_course.get(cid, [])
        if c_tasks:
            next_actions_scored.append(
                (
                    score + 1.5,
                    {
                        "text": f"Before class task: {c_tasks[0]['title']}",
                        "href": f"/courses/{cid}",
                        "sub": f"{dt.strftime('%a %H:%M')} · linked task",
                        "kind": "before_class_task",
                    },
                )
            )
        break

    # Follow-up after recent class blocks
    recent_inst = _expand_instances(schedule, today - timedelta(days=1), 2)
    for dt, row in recent_inst:
        if dt > now:
            continue
        if now - dt > timedelta(hours=12):
            continue
        if not _is_course_session(str(row.get("kind"))) or not row.get("course_id"):
            continue
        cid = int(row["course_id"])
        in_prog = [
            l
            for l in lectures
            if int(l["course_id"]) == cid and (l.get("study_progress") or "") == "in_progress"
        ]
        if in_prog:
            next_actions_scored.append(
                (
                    24.0,
                    {
                        "text": f"After class: finish «{in_prog[0]['title']}»",
                        "href": f"/lectures/{in_prog[0]['id']}",
                        "sub": "post-lecture follow-up",
                        "kind": "after_class",
                    },
                )
            )
        break

    # Missing deep dives only when lecture still not done and close to upcoming class
    for d in deep_rows:
        lid = int(d["lecture_id"])
        lec = next((l for l in lectures if int(l["id"]) == lid), None)
        if not lec or (lec.get("study_progress") or "") == "done":
            continue
        score = 11.0
        if any(int(r.get("course_id") or -1) == int(d["course_id"]) for _, r in upcoming_inst[:6]):
            score += 7.0
        next_actions_scored.append(
            (
                score,
                {
                    "text": f"Generate deep dive: {d['topic_title']}",
                    "href": f"/lectures/{lid}/topics/{d['slug']}",
                    "sub": f"{d.get('course_name') or ''} · {d.get('lecture_title') or ''}",
                    "kind": "deep_dive",
                },
            )
        )
    next_actions_scored.sort(key=lambda x: x[0], reverse=True)
    next_actions: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for _score, item in next_actions_scored:
        key = str(item.get("text") or "").strip().lower()
        if not key or key in seen_text:
            continue
        seen_text.add(key)
        next_actions.append(item)
        if len(next_actions) >= 5:
            break

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

    weekly_learning = build_weekly_learning_grid(today, schedule, lectures)

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "weekday_label": WEEKDAY_NAMES[today.weekday()],
        "today_iso": today.isoformat(),
        "now_lines": now_lines,
        "today_schedule": [
            {
                "text": f"{r['start_time']}–{r['end_time']} · {r['title']}",
                "href": _course_href(r),
                "sub": _planner_kind_display(str(r.get("kind") or ""))
                + (f" · {r.get('course_name')}" if r.get("course_name") else ""),
            }
            for r in today_rows
        ],
        "today_study": today_study[:10],
        "tomorrow_schedule": [
            {
                "text": f"{r['start_time']}–{r['end_time']} · {r['title']}",
                "href": _course_href(r),
                "sub": _planner_kind_display(str(r.get("kind") or ""))
                + (f" · {r.get('course_name')}" if r.get("course_name") else ""),
            }
            for r in tomorrow_rows
        ],
        "tomorrow_study": tomorrow_study[:8],
        "focus_lines": focus_lines,
        "next_actions": next_actions,
        "course_attention": course_attention[:8],
        "deep_dive_by_course": deep_dive_by_course,
        "next_up": next_up,
        "catch_up": catch_up[:16],
        "deep_dive_lines": deep_dive_lines,
        "stats_line": f"In progress: {n_ip} · Not started: {n_ns} · Done: {n_done}",
        "schedule_items": schedule,
        "today_rows_raw": today_rows,
        "weekly_learning": weekly_learning,
    }


def _course_href(row: dict[str, Any]) -> str | None:
    cid = row.get("course_id")
    if cid is None:
        return None
    try:
        return f"/courses/{int(cid)}"
    except (TypeError, ValueError):
        return None
