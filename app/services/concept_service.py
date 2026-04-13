"""Concepts and lecture_concepts persistence."""

from __future__ import annotations

from typing import Any, Optional

from app.db.database import get_connection
from app.services.concept_normalize import normalize_concept_key
from app.services.concept_quality import filter_concept_rows_for_display


def _get_or_create_concept_id(conn, display_name: str, norm: str) -> int:
    cur = conn.execute("SELECT id FROM concepts WHERE normalized_name = ?", (norm,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur = conn.execute(
        "INSERT INTO concepts (name, normalized_name) VALUES (?, ?)",
        (display_name, norm),
    )
    return int(cur.lastrowid)


def replace_lecture_concepts(lecture_id: int, display_names: list[str]) -> tuple[bool, Optional[str]]:
    """
    Replace all lecture_concepts for this lecture with links to concepts
    (reused by normalized_name). Idempotent for regeneration.
    """
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM lecture_concepts WHERE lecture_id = ?", (lecture_id,))
            seen_norm: set[str] = set()
            for raw in display_names:
                norm = normalize_concept_key(raw)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                disp = raw.strip()
                if len(disp) > 200:
                    disp = disp[:197] + "…"
                cid = _get_or_create_concept_id(conn, disp, norm)
                conn.execute(
                    """
                    INSERT INTO lecture_concepts (lecture_id, concept_id, relevance_score)
                    VALUES (?, ?, NULL)
                    """,
                    (lecture_id, cid),
                )
            conn.commit()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def list_concepts_for_lecture(lecture_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT c.name, c.normalized_name
            FROM concepts c
            JOIN lecture_concepts lc ON lc.concept_id = c.id
            WHERE lc.lecture_id = ?
            ORDER BY c.name COLLATE NOCASE
            """,
            (lecture_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def lecture_concepts_ui_context(lecture_id: int) -> dict[str, Any]:
    """
    Concepts for the lecture detail page: filtered, capped, with counts for empty/capped hints.
    """
    rows = list_concepts_for_lecture(lecture_id)
    filtered, total, all_filtered, hit_cap = filter_concept_rows_for_display(rows)
    return {
        "items": filtered,
        "total_stored": total,
        "shown": len(filtered),
        "hit_display_cap": hit_cap,
        "all_filtered_out": all_filtered,
    }
