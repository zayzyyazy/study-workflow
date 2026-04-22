"""Short-lived server-side tokens for two-step PDF intake (local single-user app)."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from threading import Lock

_TTL_SEC = 900.0
_store: dict[str, tuple[float, Path, str]] = {}
_lock = Lock()


def store_file(path: Path, original_filename: str) -> str:
    token = secrets.token_urlsafe(20)
    safe_name = (original_filename or path.name).strip() or path.name
    with _lock:
        _purge_locked()
        _store[token] = (time.time(), path, safe_name)
    return token


def pop_entry(token: str) -> tuple[Path, str] | None:
    """Remove token from store and return (temp path, original upload filename)."""
    with _lock:
        _purge_locked()
        entry = _store.pop((token or "").strip(), None)
        if not entry:
            return None
        return entry[1], entry[2]


def _purge_locked() -> None:
    now = time.time()
    dead: list[str] = []
    for k, (ts, _, _) in _store.items():
        if now - ts > _TTL_SEC:
            dead.append(k)
    for k in dead:
        tup = _store.pop(k, None)
        if tup:
            try:
                tup[1].unlink(missing_ok=True)
            except OSError:
                pass
