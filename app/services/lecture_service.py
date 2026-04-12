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


def count_lectures() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM lectures")
        return int(cur.fetchone()[0])


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
                   c.id AS course_id, c.name AS course_name
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
                   c.id AS course_id, c.name AS course_name
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
        SELECT id, course_id, title, slug, source_file_name, status, created_at
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
            SELECT l.id, l.title, l.slug, l.status, l.created_at,
                   l.source_file_name, c.id AS course_id, c.name AS course_name
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
            SELECT id, course_id, title, slug, source_file_name, status, created_at
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
                   l.source_file_path, l.extracted_text_path, l.status, l.created_at,
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
