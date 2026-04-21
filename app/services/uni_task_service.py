"""Small uni task layer with deterministic linking + optional AI fallback."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

from app.db.database import get_connection
from app.services import lecture_service
from app.services.openai_service import chat_completion_markdown, is_openai_configured

_STOPWORDS = {
    "the",
    "for",
    "and",
    "mit",
    "und",
    "fur",
    "vor",
    "next",
    "before",
    "lecture",
    "vorlesung",
    "class",
    "tomorrow",
    "today",
    "finish",
    "review",
    "prepare",
    "read",
    "do",
    "work",
    "on",
}


def _normalize_text(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ß", "ss")
    return s


def _tokens(raw: str) -> set[str]:
    bits = re.findall(r"[a-z0-9]{2,}", _normalize_text(raw))
    return {b for b in bits if b not in _STOPWORDS}


def _guess_kind(task_title: str) -> str | None:
    t = _normalize_text(task_title)
    if any(k in t for k in ("ubungsblatt", "uebungsblatt", "exercise", "blatt")):
        return "exercise"
    if any(k in t for k in ("review", "revisit", "wiederholen", "deep dive")):
        return "review"
    if any(k in t for k in ("read", "lesen", "skim")):
        return "read"
    if any(k in t for k in ("prepare", "prep", "vorbereiten")):
        return "prepare"
    if "project" in t or "milestone" in t or "presentation" in t:
        return "project"
    return None


def _extract_lecture_number(task_title: str) -> int | None:
    low = _normalize_text(task_title)
    m = re.search(r"(?:lecture|vorlesung|lec)\s*0*(\d{1,2})", low)
    if m:
        return int(m.group(1))
    return None


def _extract_topic_hint(task_title: str, matched_lecture_title: str | None) -> str | None:
    task_toks = _tokens(task_title)
    lec_toks = _tokens(matched_lecture_title or "")
    shared = [t for t in sorted(task_toks) if t in lec_toks and len(t) >= 5]
    if shared:
        return ", ".join(shared[:3])
    return None


def _deterministic_link(task_title: str) -> dict[str, Any]:
    lectures = lecture_service.list_lectures_for_planner()
    task_toks = _tokens(task_title)
    lecture_num = _extract_lecture_number(task_title)
    best_course_id: int | None = None
    best_course_name: str | None = None
    best_course_score = 0.0
    best_lecture: dict[str, Any] | None = None
    best_lecture_score = 0.0

    for lec in lectures:
        cname = str(lec.get("course_name") or "")
        ltitle = str(lec.get("title") or "")
        cscore = len(task_toks & _tokens(cname))
        lscore = len(task_toks & _tokens(ltitle))
        if lecture_num is not None:
            title_norm = _normalize_text(ltitle)
            if re.search(rf"(?:lecture|vorlesung|lec)\s*0*{lecture_num}\b", title_norm):
                lscore += 4
            if re.search(rf"\b0*{lecture_num}\b", title_norm):
                lscore += 1
        if cscore > best_course_score:
            best_course_score = float(cscore)
            best_course_id = int(lec["course_id"])
            best_course_name = cname
        total = lscore + (0.4 * cscore)
        if total > best_lecture_score:
            best_lecture_score = float(total)
            best_lecture = lec

    linked_course_id = best_course_id if best_course_score >= 1 else None
    linked_lecture_id = None
    linked_lecture_title = None
    if best_lecture is not None and best_lecture_score >= 1.5:
        linked_lecture_id = int(best_lecture["id"])
        linked_lecture_title = str(best_lecture["title"])
        if linked_course_id is None:
            linked_course_id = int(best_lecture["course_id"])
            best_course_name = str(best_lecture.get("course_name") or "")

    confidence = min(0.95, (best_course_score * 0.18) + (best_lecture_score * 0.11))
    return {
        "course_id": linked_course_id,
        "course_name": best_course_name,
        "lecture_id": linked_lecture_id,
        "lecture_title": linked_lecture_title,
        "topic": _extract_topic_hint(task_title, linked_lecture_title),
        "kind": _guess_kind(task_title),
        "confidence": round(confidence, 2),
        "source": "deterministic",
    }


def _ai_fallback_link(task_title: str, deterministic: dict[str, Any]) -> dict[str, Any]:
    if not is_openai_configured():
        return deterministic
    lectures = lecture_service.list_lectures_for_planner()
    catalog = [
        {
            "lecture_id": int(l["id"]),
            "course_id": int(l["course_id"]),
            "course_name": str(l.get("course_name") or ""),
            "lecture_title": str(l.get("title") or ""),
        }
        for l in lectures[:300]
    ]
    prompt = (
        "Classify this uni task and suggest a link.\n"
        "Return JSON only with keys: task_kind, course_id, lecture_id, topic, confidence.\n"
        "Use null when unknown.\n"
        f"Task: {task_title}\n"
        f"Deterministic_guess: {json.dumps(deterministic, ensure_ascii=True)}\n"
        f"Lecture_catalog: {json.dumps(catalog, ensure_ascii=True)}"
    )
    ok, text, _err = chat_completion_markdown(
        system_prompt="You are a strict JSON classifier. No prose.",
        user_prompt=prompt,
        max_tokens=220,
    )
    if not ok:
        return deterministic
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return deterministic
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return deterministic
    course_id = parsed.get("course_id")
    lecture_id = parsed.get("lecture_id")
    confidence = float(parsed.get("confidence") or 0.0)
    if confidence < 0.45:
        return deterministic
    return {
        "course_id": int(course_id) if course_id else deterministic.get("course_id"),
        "course_name": deterministic.get("course_name"),
        "lecture_id": int(lecture_id) if lecture_id else deterministic.get("lecture_id"),
        "lecture_title": deterministic.get("lecture_title"),
        "topic": (parsed.get("topic") or deterministic.get("topic") or None),
        "kind": (parsed.get("task_kind") or deterministic.get("kind") or None),
        "confidence": round(confidence, 2),
        "source": "ai",
    }


def _infer_due_date(raw_due_date: str | None, title: str) -> str | None:
    if raw_due_date and raw_due_date.strip():
        return raw_due_date.strip()
    text = _normalize_text(title)
    today = date.today()
    if "tomorrow" in text:
        return (today + timedelta(days=1)).isoformat()
    if "today" in text:
        return today.isoformat()
    return None


def _hydrate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        due = str(r.get("due_date") or "").strip()
        if due:
            try:
                d = date.fromisoformat(due)
                delta = (d - date.today()).days
                if delta < 0:
                    due_label = f"Overdue by {abs(delta)}d"
                elif delta == 0:
                    due_label = "Due today"
                elif delta == 1:
                    due_label = "Due tomorrow"
                else:
                    due_label = f"Due in {delta}d"
            except ValueError:
                due_label = due
        else:
            due_label = ""
        r["due_label"] = due_label
        out.append(r)
    return out


def create_task(title: str, due_date: str | None = None) -> tuple[bool, str]:
    clean = (title or "").strip()
    if not clean:
        return False, "Task title is required."
    deterministic = _deterministic_link(clean)
    linked = deterministic
    if (deterministic.get("confidence") or 0) < 0.55:
        linked = _ai_fallback_link(clean, deterministic)
    final_due = _infer_due_date(due_date, clean)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO uni_tasks (
                title, status, due_date, task_kind, course_id, lecture_id,
                linked_topic, link_source, link_confidence, updated_at
            )
            VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                clean,
                final_due,
                linked.get("kind"),
                linked.get("course_id"),
                linked.get("lecture_id"),
                linked.get("topic"),
                linked.get("source"),
                linked.get("confidence"),
            ),
        )
        conn.commit()
    return True, "Task added."


def list_tasks(*, status: str = "open", limit: int = 20) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if status in {"open", "done"}:
        where = "WHERE t.status = ?"
        params.append(status)
    params.append(limit)
    with get_connection() as conn:
        cur = conn.execute(
            f"""
            SELECT t.id, t.title, t.status, t.due_date, t.task_kind, t.linked_topic,
                   t.link_source, t.link_confidence, t.course_id, t.lecture_id,
                   t.created_at, t.updated_at,
                   c.name AS course_name,
                   l.title AS lecture_title
            FROM uni_tasks t
            LEFT JOIN courses c ON c.id = t.course_id
            LEFT JOIN lectures l ON l.id = t.lecture_id
            {where}
            ORDER BY
                CASE WHEN t.due_date IS NULL OR t.due_date = '' THEN 1 ELSE 0 END,
                t.due_date ASC,
                t.created_at DESC
            LIMIT ?
            """,
            params,
        )
        return _hydrate([dict(row) for row in cur.fetchall()])


def set_done(task_id: int, done: bool) -> tuple[bool, str]:
    new_status = "done" if done else "open"
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE uni_tasks SET status = ?, updated_at = datetime('now') WHERE id = ?",
            (new_status, task_id),
        )
        conn.commit()
    if cur.rowcount <= 0:
        return False, "Task not found."
    return True, "Task updated."


def delete_task(task_id: int) -> tuple[bool, str]:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM uni_tasks WHERE id = ?", (task_id,))
        conn.commit()
    if cur.rowcount <= 0:
        return False, "Task not found."
    return True, "Task deleted."


def update_task(task_id: int, title: str, due_date: str | None = None) -> tuple[bool, str]:
    clean = (title or "").strip()
    if not clean:
        return False, "Task title is required."
    deterministic = _deterministic_link(clean)
    linked = deterministic
    if (deterministic.get("confidence") or 0) < 0.55:
        linked = _ai_fallback_link(clean, deterministic)
    final_due = _infer_due_date(due_date, clean)
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE uni_tasks
            SET title = ?, due_date = ?, task_kind = ?, course_id = ?, lecture_id = ?,
                linked_topic = ?, link_source = ?, link_confidence = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                clean,
                final_due,
                linked.get("kind"),
                linked.get("course_id"),
                linked.get("lecture_id"),
                linked.get("topic"),
                linked.get("source"),
                linked.get("confidence"),
                task_id,
            ),
        )
        conn.commit()
    if cur.rowcount <= 0:
        return False, "Task not found."
    return True, "Task saved."
