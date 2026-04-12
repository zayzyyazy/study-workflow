"""Course CRUD and listing."""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from app.db.database import get_connection
from app.services.slugs import slugify


def count_courses() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM courses")
        return int(cur.fetchone()[0])


def list_courses() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, slug, created_at FROM courses ORDER BY name COLLATE NOCASE"
        )
        return [dict(row) for row in cur.fetchall()]


def get_course_by_id(course_id: int) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, slug, created_at FROM courses WHERE id = ?",
            (course_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_course_by_slug(slug: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, slug, created_at FROM courses WHERE slug = ?",
            (slug,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _unique_course_slug(conn: sqlite3.Connection, base_slug: str) -> str:
    slug = base_slug
    n = 2
    while True:
        cur = conn.execute("SELECT 1 FROM courses WHERE slug = ?", (slug,))
        if cur.fetchone() is None:
            return slug
        slug = f"{base_slug}-{n}"
        n += 1


def create_course(name: str) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ValueError("Course name is required.")
    base = slugify(name)
    with get_connection() as conn:
        slug = _unique_course_slug(conn, base)
        cur = conn.execute(
            "INSERT INTO courses (name, slug) VALUES (?, ?)",
            (name, slug),
        )
        conn.commit()
        cid = cur.lastrowid
    return get_course_by_id(cid)  # type: ignore[return-value]
