"""Lecture CRUD, recent list, and per-course listing."""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from app.db.database import get_connection
from app.services.slugs import slugify

KNOWN_LECTURE_STATUSES = (
    "uploaded",
    "text_extracted",
    "extraction_failed",
    "ready_for_generation",
    "generation_pending",
    "generation_complete",
    "generation_failed",
)

# User study progress (separate from pipeline `status` above)
STUDY_PROGRESS_STATES = ("not_started", "in_progress", "done")


def count_lectures() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM lectures")
        return int(cur.fetchone()[0])


def count_lectures_for_course(course_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM lectures WHERE course_id = ?",
            (course_id,),
        )
        return int(cur.fetchone()[0])


def count_study_progress_in_course(course_id: int, progress: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM lectures
            WHERE course_id = ? AND study_progress = ?
            """,
            (course_id, progress),
        )
        return int(cur.fetchone()[0])


def study_progress_library_totals() -> dict[str, int]:
    """Total lectures and how many marked done (for home summary)."""
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM lectures")
        total = int(cur.fetchone()[0])
        cur = conn.execute("SELECT COUNT(*) FROM lectures WHERE study_progress = 'done'")
        done = int(cur.fetchone()[0])
        return {"total": total, "done": done}


def count_lectures_by_status() -> dict[str, int]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM lectures GROUP BY status ORDER BY status"
        )
        return {str(row[0]): int(row[1]) for row in cur.fetchall()}


def list_lectures_needing_attention(limit: int = 25) -> list[dict[str, Any]]:
    """Lectures that likely need a follow-up action."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title, l.status, l.created_at,
                   l.source_file_path,
                   c.id AS course_id, c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            WHERE l.status IN ('extraction_failed', 'generation_failed', 'ready_for_generation')
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def search_lectures_global(q: str, limit: int = 50) -> list[dict[str, Any]]:
    """Match lecture title or course name (case-insensitive substring)."""
    needle = (q or "").strip()
    if not needle:
        return []
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title, l.status, l.created_at,
                   l.source_file_path,
                   c.id AS course_id, c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            WHERE instr(lower(l.title), lower(?)) > 0
               OR instr(lower(c.name), lower(?)) > 0
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (needle, needle, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def list_lectures_for_course_filtered(
    course_id: int,
    *,
    title_query: str = "",
    status: str = "",
) -> list[dict[str, Any]]:
    """Filter lectures by optional title substring and/or exact status."""
    tq = (title_query or "").strip()
    st = (status or "").strip()
    if st and st not in KNOWN_LECTURE_STATUSES:
        st = ""

    conditions = ["course_id = ?"]
    params: list[Any] = [course_id]
    if tq:
        conditions.append("instr(lower(title), lower(?)) > 0")
        params.append(tq)
    if st:
        conditions.append("status = ?")
        params.append(st)

    sql = f"""
        SELECT id, course_id, title, slug, source_file_name, source_file_path, status, study_progress, is_starred, created_at
        FROM lectures
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
    """
    with get_connection() as conn:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def list_recent_lectures(limit: int = 10) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title, l.slug, l.status, l.study_progress, l.is_starred, l.created_at,
                   l.source_file_name, l.source_file_path,
                   c.id AS course_id, c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def list_lectures_for_course(course_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, course_id, title, slug, source_file_name, source_file_path, status, study_progress, is_starred, created_at
            FROM lectures
            WHERE course_id = ?
            ORDER BY created_at DESC
            """,
            (course_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_lecture_by_id(lecture_id: int) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.course_id, l.title, l.slug, l.source_file_name,
                   l.source_file_path, l.extracted_text_path, l.status, l.study_progress, l.is_starred, l.created_at,
                   c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            WHERE l.id = ?
            """,
            (lecture_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _next_lecture_index(conn: sqlite3.Connection, course_id: int) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM lectures WHERE course_id = ?",
        (course_id,),
    )
    count = cur.fetchone()[0]
    return int(count) + 1


def _unique_lecture_slug(conn: sqlite3.Connection, course_id: int, base_slug: str) -> str:
    slug = base_slug
    n = 2
    while True:
        cur = conn.execute(
            "SELECT 1 FROM lectures WHERE course_id = ? AND slug = ?",
            (course_id, slug),
        )
        if cur.fetchone() is None:
            return slug
        slug = f"{base_slug}-{n}"
        n += 1


def insert_lecture(
    course_id: int,
    title: str,
    source_file_name: str,
    source_file_path: str,
    extracted_text_path: Optional[str],
    status: str,
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ValueError("Lecture title is required.")
    base = slugify(title)
    with get_connection() as conn:
        slug = _unique_lecture_slug(conn, course_id, base)
        cur = conn.execute(
            """
            INSERT INTO lectures (
                course_id, title, slug, source_file_name, source_file_path,
                extracted_text_path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                course_id,
                title,
                slug,
                source_file_name,
                source_file_path,
                extracted_text_path,
                status,
            ),
        )
        conn.commit()
        lid = cur.lastrowid
    return get_lecture_by_id(lid)  # type: ignore[return-value]


def lecture_index_for_course(course_id: int) -> int:
    """1-based display index for the next lecture in this course."""
    with get_connection() as conn:
        return _next_lecture_index(conn, course_id)


def set_lecture_source_and_extraction(
    lecture_id: int,
    *,
    source_file_name: str,
    source_file_path: str,
    extracted_text_path: Optional[str],
    status: str,
) -> None:
    """Sets source fields and extraction outcome (extracted_text_path may be NULL)."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE lectures SET
                source_file_name = ?,
                source_file_path = ?,
                extracted_text_path = ?,
                status = ?
            WHERE id = ?
            """,
            (source_file_name, source_file_path, extracted_text_path, status, lecture_id),
        )
        conn.commit()


def delete_lecture_row(lecture_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM lectures WHERE id = ?", (lecture_id,))
        conn.commit()


def update_lecture_status(lecture_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE lectures SET status = ? WHERE id = ?", (status, lecture_id))
        conn.commit()


def set_lecture_study_progress(lecture_id: int, progress: str) -> bool:
    """Set user study progress. Returns False if invalid."""
    if progress not in STUDY_PROGRESS_STATES:
        return False
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE lectures SET study_progress = ? WHERE id = ?",
            (progress, lecture_id),
        )
        conn.commit()
        return cur.rowcount > 0


def reset_all_study_progress() -> int:
    """Set every lecture's study_progress to not_started. Returns rows updated."""
    with get_connection() as conn:
        cur = conn.execute("UPDATE lectures SET study_progress = 'not_started'")
        conn.commit()
        return int(cur.rowcount)


def reset_study_progress_for_course(course_id: int) -> int:
    """Reset study progress for lectures in one course."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE lectures SET study_progress = 'not_started' WHERE course_id = ?",
            (course_id,),
        )
        conn.commit()
        return int(cur.rowcount)


def reset_single_lecture_study_progress(lecture_id: int) -> bool:
    """Set one lecture's study_progress to not_started. Does not touch pipeline status."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE lectures SET study_progress = 'not_started' WHERE id = ?",
            (lecture_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def set_lecture_starred(lecture_id: int, starred: bool) -> bool:
    """Set is_starred (0/1)."""
    v = 1 if starred else 0
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE lectures SET is_starred = ? WHERE id = ?",
            (v, lecture_id),
        )
        conn.commit()
        return cur.rowcount > 0


def reset_lecture_user_flags(lecture_id: int) -> bool:
    """Reset study progress and un-star one lecture."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE lectures
            SET study_progress = 'not_started', is_starred = 0
            WHERE id = ?
            """,
            (lecture_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def list_starred_lectures(limit: int = 24) -> list[dict[str, Any]]:
    """Starred lectures for home (priority list)."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title, l.status, l.study_progress, l.is_starred, l.created_at,
                   l.source_file_path,
                   c.id AS course_id, c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            WHERE l.is_starred = 1
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def list_lectures_for_planner() -> list[dict[str, Any]]:
    """All lectures with course names for deterministic planner views."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title, l.slug, l.status, l.study_progress, l.is_starred, l.source_file_path,
                   l.created_at,
                   c.id AS course_id, c.name AS course_name, c.slug AS course_slug
            FROM lectures l
            JOIN courses c ON c.id = l.course_id
            ORDER BY c.name COLLATE NOCASE, l.title COLLATE NOCASE
            """
        )
        return [dict(row) for row in cur.fetchall()]
