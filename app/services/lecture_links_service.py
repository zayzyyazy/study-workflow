"""Course-internal lecture linking (deterministic prev/next + concept overlap)."""

from __future__ import annotations

import re
from typing import Any, Optional

from app.db.database import get_connection
from app.services import lecture_service

_DISPLAY_PREFIX = re.compile(r"^lecture\s+\d+\s*-\s*", re.I)


def short_lecture_label(title: str, *, max_len: int = 52) -> str:
    s = (title or "").strip()
    s = _DISPLAY_PREFIX.sub("", s).strip()
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def _title_is_readable(title: str) -> bool:
    """Skip linking hints when the stored title looks like noise (e.g. digits-only scrap)."""
    s = short_lecture_label(str(title), max_len=120)
    letters = sum(1 for c in s if c.isalpha())
    if letters < 4:
        return False
    digits = sum(1 for c in s if c.isdigit())
    if digits >= 6 and digits > letters:
        return False
    return True


def _position_in_sequence(seq: list[dict[str, Any]], lecture_id: int) -> int:
    for i, row in enumerate(seq):
        if int(row["id"]) == lecture_id:
            return i
    return -1


def concept_neighbor_lectures(lecture_id: int, *, limit: int = 4) -> list[dict[str, Any]]:
    """Other lectures in the same course that share indexed concepts (2+ overlaps)."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT l2.id AS id, l2.title AS title, COUNT(*) AS shared
            FROM lecture_concepts lc1
            JOIN lecture_concepts lc2
              ON lc1.concept_id = lc2.concept_id AND lc1.lecture_id != lc2.lecture_id
            JOIN lectures l1 ON l1.id = lc1.lecture_id
            JOIN lectures l2 ON l2.id = lc2.lecture_id AND l2.course_id = l1.course_id
            WHERE lc1.lecture_id = ?
            GROUP BY l2.id
            HAVING COUNT(*) >= 2
            ORDER BY shared DESC, l2.id ASC
            LIMIT ?
            """,
            (lecture_id, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def build_lecture_links(lecture_id: int) -> dict[str, Any]:
    """
    Compact context for the lecture page: prev/next in course + concept neighbors.
    """
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return {"lines": [], "prev": None, "next": None}

    cid = int(lec["course_id"])
    seq = lecture_service.list_lectures_course_sequence(cid)
    pos = _position_in_sequence(seq, lecture_id)
    prev_row: Optional[dict[str, Any]] = None
    next_row: Optional[dict[str, Any]] = None
    if pos >= 0:
        if pos > 0:
            prev_row = dict(seq[pos - 1])
        if pos < len(seq) - 1:
            next_row = dict(seq[pos + 1])

    lines: list[dict[str, Any]] = []
    if prev_row and _title_is_readable(str(prev_row.get("title") or "")):
        lines.append(
            {
                "text": f"Builds on «{short_lecture_label(str(prev_row['title']))}»",
                "href": f"/lectures/{prev_row['id']}",
                "sub": "earlier in course",
            }
        )
    if next_row and _title_is_readable(str(next_row.get("title") or "")):
        lines.append(
            {
                "text": f"Next in course: «{short_lecture_label(str(next_row['title']))}»",
                "href": f"/lectures/{next_row['id']}",
                "sub": "follow-up",
            }
        )

    for nb in concept_neighbor_lectures(lecture_id, limit=3):
        oid = int(nb["id"])
        if prev_row and oid == int(prev_row["id"]):
            continue
        if next_row and oid == int(next_row["id"]):
            continue
        if not _title_is_readable(str(nb.get("title") or "")):
            continue
        n = int(nb.get("shared") or 0)
        lines.append(
            {
                "text": f"Related: «{short_lecture_label(str(nb['title']))}»",
                "href": f"/lectures/{oid}",
                "sub": f"{n} shared topics" if n else "concept overlap",
            }
        )

    return {
        "prev": prev_row,
        "next": next_row,
        "lines": lines[:6],
    }


def home_connection_hints(*, limit: int = 5) -> list[dict[str, Any]]:
    """
    Short library-aware lines for the home dashboard (no AI).
    """
    lectures = lecture_service.list_lectures_for_planner()
    hints: list[dict[str, Any]] = []
    seen_href: set[str] = set()

    def add(text: str, href: str, sub: str) -> None:
        if href in seen_href or len(hints) >= limit:
            return
        seen_href.add(href)
        hints.append({"text": text, "href": href, "sub": sub})

    in_progress = [l for l in lectures if l.get("study_progress") == "in_progress"]
    in_progress.sort(
        key=lambda x: (-(int(x.get("is_starred") or 0)), x.get("created_at") or ""),
    )

    for l in in_progress[:4]:
        lid = int(l["id"])
        cid = int(l["course_id"])
        seq = lecture_service.list_lectures_course_sequence(cid)
        pos = _position_in_sequence(seq, lid)
        if pos > 0:
            prev = seq[pos - 1]
            if (prev.get("study_progress") or "") != "done" and _title_is_readable(
                str(prev.get("title") or "")
            ):
                add(
                    f"Review «{short_lecture_label(str(prev['title']))}» first",
                    f"/lectures/{prev['id']}",
                    f"before «{short_lecture_label(str(l['title']))}»",
                )
        for nb in concept_neighbor_lectures(lid, limit=2):
            oid = int(nb["id"])
            if oid == lid:
                continue
            add(
                f"Related read: «{short_lecture_label(str(nb['title']))}»",
                f"/lectures/{oid}",
                "same course topics",
            )
            break

    starred = [x for x in lectures if int(x.get("is_starred") or 0) and x.get("study_progress") != "done"]
    starred.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    for l in starred[:2]:
        add(
            f"Continue starred: «{short_lecture_label(str(l['title']))}»",
            f"/lectures/{l['id']}",
            str(l.get("course_name") or ""),
        )

    return hints[:limit]
