"""Course-level concept aggregation and optional Markdown index file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config import COURSES_DIR
from app.db.database import get_connection


def aggregate_course_concepts_filtered(
    course_id: int,
    *,
    name_query: str = "",
    only_concept_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Course concepts with optional name substring filter and/or single-concept focus.
    """
    rows = aggregate_course_concepts(course_id)
    if only_concept_id is not None:
        rows = [r for r in rows if int(r["concept_id"]) == only_concept_id]
    nq = (name_query or "").strip().lower()
    if nq:
        out: list[dict[str, Any]] = []
        for r in rows:
            name = (r.get("name") or "").lower()
            norm = (r.get("normalized_name") or "").lower()
            if nq in name or nq in norm:
                out.append(r)
        rows = out
    return rows


def aggregate_course_concepts(course_id: int) -> list[dict[str, Any]]:
    """
    Recurring concepts: lecture count and linked lectures.
    Sorted by lecture count desc, then name.
    """
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT c.id AS concept_id, c.name, c.normalized_name,
                   COUNT(DISTINCT lc.lecture_id) AS lecture_count
            FROM concepts c
            JOIN lecture_concepts lc ON lc.concept_id = c.id
            JOIN lectures l ON l.id = lc.lecture_id
            WHERE l.course_id = ?
            GROUP BY c.id
            ORDER BY lecture_count DESC, c.name COLLATE NOCASE
            """,
            (course_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    for row in rows:
        row["lectures"] = _lectures_for_concept(course_id, int(row["concept_id"]))
    return rows


def _lectures_for_concept(course_id: int, concept_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l.id, l.title
            FROM lectures l
            JOIN lecture_concepts lc ON lc.lecture_id = l.id
            WHERE l.course_id = ? AND lc.concept_id = ?
            ORDER BY l.title COLLATE NOCASE
            """,
            (course_id, concept_id),
        )
        return [dict(r) for r in cur.fetchall()]


def write_course_concept_index_file(course_slug: str, course_name: str, course_id: int) -> Optional[Path]:
    """
    courses/<slug>/course_index/concept_index.md — regenerated whenever a lecture is re-indexed.
    """
    agg = aggregate_course_concepts(course_id)
    lines = [
        f"# Concept index — {course_name}",
        "",
        f"_Auto-generated from indexed lectures. Concepts appear after study materials are generated._",
        "",
    ]
    if not agg:
        lines.append("_No concepts indexed yet._")
    else:
        for row in agg:
            cnt = row["lecture_count"]
            name = row["name"]
            lecs = row["lectures"]
            lines.append(f"## {name} ({cnt} lecture{'s' if cnt != 1 else ''})")
            for lec in lecs:
                lines.append(f"- [{lec['title']}](/lectures/{lec['id']})")
            lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    out_dir = COURSES_DIR / course_slug / "course_index"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "concept_index.md"
        path.write_text(body, encoding="utf-8")
        return path
    except OSError:
        return None
