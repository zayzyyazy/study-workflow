"""Local degree / credit-point progress tracker (simple CP totals, not a transcript)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from app.db.database import get_connection


def get_target_cp() -> float:
    with get_connection() as conn:
        cur = conn.execute("SELECT target_cp FROM degree_progress_meta WHERE id = 1")
        row = cur.fetchone()
        if not row:
            conn.execute("INSERT OR IGNORE INTO degree_progress_meta (id, target_cp) VALUES (1, 180)")
            conn.commit()
            return 180.0
        return float(row[0] or 180)


def set_target_cp(value: float) -> tuple[bool, str]:
    if value <= 0 or value > 600:
        return False, "Target CP must be between 1 and 600."
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO degree_progress_meta (id, target_cp) VALUES (1, 180)")
        conn.execute("UPDATE degree_progress_meta SET target_cp = ? WHERE id = 1", (value,))
        conn.commit()
    return True, "Degree target updated."


def list_entries() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, title, cp, done, category, sort_order, created_at
            FROM degree_progress_entries
            ORDER BY (category IS NULL), category COLLATE NOCASE, sort_order ASC, id ASC
            """
        )
        return [dict(r) for r in cur.fetchall()]


def _next_sort_order(conn) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM degree_progress_entries")
    return int(cur.fetchone()[0])


def add_entry(title: str, cp: float, category: str | None = None) -> tuple[bool, str]:
    t = (title or "").strip()
    if not t:
        return False, "Title is required."
    if cp <= 0 or cp > 120:
        return False, "CP must be positive (max 120 per entry)."
    cat = (category or "").strip() or None
    with get_connection() as conn:
        so = _next_sort_order(conn)
        conn.execute(
            """
            INSERT INTO degree_progress_entries (title, cp, done, category, sort_order)
            VALUES (?, ?, 0, ?, ?)
            """,
            (t, float(cp), cat, so),
        )
        conn.commit()
    return True, "Entry added."


def set_done(entry_id: int, done: bool) -> tuple[bool, str]:
    v = 1 if done else 0
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE degree_progress_entries SET done = ? WHERE id = ?",
            (v, entry_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        return False, "Entry not found."
    return True, "Updated."


def delete_entry(entry_id: int) -> tuple[bool, str]:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM degree_progress_entries WHERE id = ?", (entry_id,))
        conn.commit()
    if cur.rowcount == 0:
        return False, "Entry not found."
    return True, "Removed."


def update_entry(
    entry_id: int,
    *,
    title: str,
    cp: float,
    category: str | None = None,
) -> tuple[bool, str]:
    t = (title or "").strip()
    if not t:
        return False, "Title is required."
    if cp <= 0 or cp > 120:
        return False, "CP must be positive (max 120 per entry)."
    cat = (category or "").strip() or None
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE degree_progress_entries
            SET title = ?, cp = ?, category = ?
            WHERE id = ?
            """,
            (t, float(cp), cat, entry_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        return False, "Entry not found."
    return True, "Saved."


def summarize() -> dict[str, Any]:
    """Totals for dashboard and full page."""
    target = get_target_cp()
    entries = list_entries()
    total_listed = sum(float(e["cp"]) for e in entries)
    done_cp = sum(float(e["cp"]) for e in entries if int(e.get("done") or 0))
    open_cp = sum(float(e["cp"]) for e in entries if not int(e.get("done") or 0))
    n_done = sum(1 for e in entries if int(e.get("done") or 0))
    n_open = len(entries) - n_done
    remaining_to_goal = max(0.0, float(target) - done_cp)
    pct = (done_cp / target * 100.0) if target > 0 else 0.0
    pct = min(100.0, pct)
    return {
        "target_cp": target,
        "total_listed_cp": round(total_listed, 2),
        "done_cp": round(done_cp, 2),
        "open_cp": round(open_cp, 2),
        "remaining_to_goal_cp": round(remaining_to_goal, 2),
        "pct_of_target": round(pct, 1),
        "entry_count": len(entries),
        "open_count": n_open,
        "done_count": n_done,
    }


def entries_by_category() -> list[tuple[str, list[dict[str, Any]]]]:
    entries = list_entries()
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        key = (e.get("category") or "").strip() or "General"
        by_cat[key].append(e)
    # stable: General first, then alpha
    keys = sorted(by_cat.keys(), key=lambda k: (0 if k == "General" else 1, k.lower()))
    return [(k, by_cat[k]) for k in keys]
