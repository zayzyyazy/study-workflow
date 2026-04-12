"""Database lifecycle (called from app startup)."""

from app.db.database import init_db


def initialize_database() -> None:
    init_db()
