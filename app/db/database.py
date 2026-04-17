"""SQLite setup: schema creation and connection helper."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.config import DATABASE_PATH, ensure_directories

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lectures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    source_file_name TEXT NOT NULL,
    source_file_path TEXT NOT NULL,
    extracted_text_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'uploaded',
    UNIQUE(course_id, slug)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id INTEGER NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lecture_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lecture_id INTEGER NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relevance_score REAL,
    UNIQUE(lecture_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_lectures_course ON lectures(course_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_lecture ON artifacts(lecture_id);
"""


def _migrate_legacy_statuses(conn: sqlite3.Connection) -> None:
    """One-time mapping for older status strings."""
    conn.execute(
        """
        UPDATE lectures SET status = 'ready_for_generation'
        WHERE status IN ('extracted', 'text_extracted')
        """
    )


def _ensure_study_progress_column(conn: sqlite3.Connection) -> None:
    """Add study_progress for user study state (not generation pipeline)."""
    cur = conn.execute("PRAGMA table_info(lectures)")
    names = {str(row[1]) for row in cur.fetchall()}
    if "study_progress" not in names:
        conn.execute(
            """
            ALTER TABLE lectures ADD COLUMN study_progress TEXT NOT NULL DEFAULT 'not_started'
            """
        )


def _ensure_is_starred_column(conn: sqlite3.Connection) -> None:
    """Add is_starred for quick priority / favorite marking (0/1)."""
    cur = conn.execute("PRAGMA table_info(lectures)")
    names = {str(row[1]) for row in cur.fetchall()}
    if "is_starred" not in names:
        conn.execute(
            """
            ALTER TABLE lectures ADD COLUMN is_starred INTEGER NOT NULL DEFAULT 0
            """
        )


def _ensure_planner_schedule_table(conn: sqlite3.Connection) -> None:
    """Weekly / one-off schedule blocks for the study planner (MVP)."""
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='planner_schedule_items'"
    )
    if cur.fetchone() is not None:
        return
    conn.execute(
        """
        CREATE TABLE planner_schedule_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER REFERENCES courses(id) ON DELETE SET NULL,
            title TEXT NOT NULL,
            kind TEXT NOT NULL,
            recurrence TEXT NOT NULL,
            weekday INTEGER,
            specific_date TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_planner_schedule_weekday ON planner_schedule_items(weekday)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_planner_schedule_date ON planner_schedule_items(specific_date)"
    )


def init_db() -> None:
    ensure_directories()
    db_path: Path = DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate_legacy_statuses(conn)
        _ensure_study_progress_column(conn)
        _ensure_is_starred_column(conn)
        _ensure_planner_schedule_table(conn)
        conn.commit()


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    ensure_directories()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()
