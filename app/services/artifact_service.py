"""Persist generated files in the artifacts table."""

from __future__ import annotations

from typing import Any

from app.db.database import get_connection

# Matches output filenames under outputs/
GENERATION_ARTIFACT_TYPES: tuple[tuple[str, str], ...] = (
    ("quick_overview", "01_quick_overview.md"),
    ("topic_map", "02_topic_map.md"),
    ("core_learning", "03_core_learning.md"),
    ("revision_sheet", "04_revision_sheet.md"),
    ("study_pack", "05_study_pack.md"),
)


def list_artifacts_for_lecture(lecture_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, lecture_id, artifact_type, file_path, created_at
            FROM artifacts
            WHERE lecture_id = ?
            ORDER BY artifact_type
            """,
            (lecture_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def delete_generation_artifacts_for_lecture(lecture_id: int) -> None:
    """Remove prior generation rows so we can insert fresh ones."""
    types = [t for t, _ in GENERATION_ARTIFACT_TYPES]
    placeholders = ",".join("?" * len(types))
    with get_connection() as conn:
        conn.execute(
            f"""
            DELETE FROM artifacts
            WHERE lecture_id = ? AND artifact_type IN ({placeholders})
            """,
            (lecture_id, *types),
        )
        conn.commit()


def insert_artifact(lecture_id: int, artifact_type: str, file_path: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (lecture_id, artifact_type, file_path)
            VALUES (?, ?, ?)
            """,
            (lecture_id, artifact_type, file_path),
        )
        conn.commit()


def replace_generation_artifacts(lecture_id: int, paths: list[tuple[str, str]]) -> None:
    """
    paths: list of (artifact_type, relative file path from app root)
    Replaces all generation artifact rows for this lecture.
    """
    delete_generation_artifacts_for_lecture(lecture_id)
    for artifact_type, rel_path in paths:
        insert_artifact(lecture_id, artifact_type, rel_path)
