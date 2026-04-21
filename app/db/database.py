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


def _ensure_topic_quiz_mistake_stats(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='topic_quiz_mistake_stats'"
    )
    if cur.fetchone() is not None:
        return
    conn.execute(
        """
        CREATE TABLE topic_quiz_mistake_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lecture_id INTEGER NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
            topic_slug TEXT NOT NULL,
            concept_key TEXT NOT NULL,
            subtopic_slug TEXT,
            wrong_count INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            last_wrong_at TEXT,
            last_right_at TEXT,
            UNIQUE(lecture_id, topic_slug, concept_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tqm_lecture_topic ON topic_quiz_mistake_stats(lecture_id, topic_slug)"
    )


def _ensure_uni_tasks_table(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='uni_tasks'")
    if cur.fetchone() is None:
        conn.execute(
            """
            CREATE TABLE uni_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                due_date TEXT,
                task_kind TEXT,
                course_id INTEGER REFERENCES courses(id) ON DELETE SET NULL,
                lecture_id INTEGER REFERENCES lectures(id) ON DELETE SET NULL,
                linked_topic TEXT,
                link_source TEXT,
                link_confidence REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
    cur = conn.execute("PRAGMA table_info(uni_tasks)")
    names = {str(row[1]) for row in cur.fetchall()}
    if "linked_topic" not in names:
        conn.execute("ALTER TABLE uni_tasks ADD COLUMN linked_topic TEXT")
    if "link_source" not in names:
        conn.execute("ALTER TABLE uni_tasks ADD COLUMN link_source TEXT")
    if "link_confidence" not in names:
        conn.execute("ALTER TABLE uni_tasks ADD COLUMN link_confidence REAL")
    if "updated_at" not in names:
        conn.execute("ALTER TABLE uni_tasks ADD COLUMN updated_at TEXT")
        conn.execute("UPDATE uni_tasks SET updated_at = created_at WHERE updated_at IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uni_tasks_status ON uni_tasks(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uni_tasks_due ON uni_tasks(due_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uni_tasks_course ON uni_tasks(course_id)")


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
        _ensure_topic_quiz_mistake_stats(conn)
        _ensure_uni_tasks_table(conn)
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
