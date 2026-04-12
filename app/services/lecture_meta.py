"""Keep meta.json aligned with SQLite and extraction state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import APP_ROOT


def read_meta(lecture_root: Path) -> dict[str, Any]:
    path = lecture_root / "meta.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_meta_payload(
    *,
    lecture_id: int,
    course_name: str,
    lecture_name: str,
    source_file_name: str,
    source_file_path: str,
    extracted_text_path: Optional[str],
    status: str,
    created_at: str,
    extraction_message: Optional[str] = None,
    generation_message: Optional[str] = None,
    generated_artifacts: Optional[list[dict[str, Any]]] = None,
    lecture_analysis: Optional[dict[str, Any]] = None,
    drop_lecture_analysis: bool = False,
    previous: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Canonical meta.json shape; merges created_at from previous file when present."""
    prev = previous or {}
    created = prev.get("created_at") or created_at
    payload: dict[str, Any] = {
        "lecture_id": lecture_id,
        "course_name": course_name,
        "lecture_name": lecture_name,
        "source_file_name": source_file_name,
        "source_file_path": source_file_path,
        "extracted_text_path": extracted_text_path,
        "status": status,
        "created_at": created,
        "updated_at": _iso_now(),
    }
    if extraction_message is not None:
        payload["extraction_message"] = extraction_message
    elif "extraction_message" in prev:
        payload["extraction_message"] = prev["extraction_message"]
    # generation_message / generated_artifacts: pass explicit "" or [] after extraction to clear stale entries
    if generation_message is not None:
        payload["generation_message"] = generation_message
    elif "generation_message" in prev:
        payload["generation_message"] = prev["generation_message"]
    if generated_artifacts is not None:
        payload["generated_artifacts"] = generated_artifacts
    elif "generated_artifacts" in prev:
        payload["generated_artifacts"] = prev["generated_artifacts"]
    if not drop_lecture_analysis:
        if lecture_analysis is not None:
            payload["lecture_analysis"] = lecture_analysis
        elif "lecture_analysis" in prev:
            payload["lecture_analysis"] = prev["lecture_analysis"]
    return payload


def write_meta(lecture_root: Path, payload: dict[str, Any]) -> Path:
    path = lecture_root / "meta.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def sync_meta_for_lecture(
    lecture_root: Path,
    *,
    lecture_id: int,
    course_name: str,
    lecture_title: str,
    source_file_name: str,
    source_rel_posix: str,
    extracted_rel_posix: Optional[str],
    status: str,
    db_created_at: str,
    extraction_message: Optional[str] = None,
    generation_message: Optional[str] = None,
    generated_artifacts: Optional[list[dict[str, Any]]] = None,
    lecture_analysis: Optional[dict[str, Any]] = None,
    drop_lecture_analysis: bool = False,
) -> None:
    prev = read_meta(lecture_root)
    payload = build_meta_payload(
        lecture_id=lecture_id,
        course_name=course_name,
        lecture_name=lecture_title,
        source_file_name=source_file_name,
        source_file_path=source_rel_posix,
        extracted_text_path=extracted_rel_posix,
        status=status,
        created_at=str(db_created_at),
        extraction_message=extraction_message,
        generation_message=generation_message,
        generated_artifacts=generated_artifacts,
        lecture_analysis=lecture_analysis,
        drop_lecture_analysis=drop_lecture_analysis,
        previous=prev,
    )
    write_meta(lecture_root, payload)


def relative_to_app(path: Path) -> str:
    try:
        return path.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
