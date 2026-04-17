"""Human-friendly storage paths for UI (display names vs on-disk layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config import APP_ROOT
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services import source_manifest


def lecture_disk_folder_name(lecture_row: dict[str, Any]) -> str:
    """Last segment of the lecture folder (e.g. ``Lecture 03 - Week 3``)."""
    sp = lecture_row.get("source_file_path") or ""
    if not sp:
        return ""
    try:
        root = lecture_root_from_source_relative(str(sp))
        return root.name
    except (OSError, ValueError):
        return ""


def enrich_lecture_rows_for_course_ui(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adds ``disk_folder_name`` for course lecture lists."""
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["disk_folder_name"] = lecture_disk_folder_name(r)
        out.append(r)
    return out


def lecture_storage_context(lecture: dict[str, Any]) -> dict[str, Any]:
    """
    Paths for the lecture detail “Files & folders” section.

    Uses stored relative paths; does not move or rename anything on disk.
    """
    course_slug = str(lecture.get("course_slug") or "")
    source_rel = (lecture.get("source_file_path") or "").replace("\\", "/")
    extracted_raw = lecture.get("extracted_text_path")
    extracted_rel = extracted_raw.replace("\\", "/") if extracted_raw else None

    lecture_root: Optional[Path] = None
    rel_root = ""
    lecture_folder_name = ""
    outputs_rel = ""
    full_root = ""
    full_course_dir = ""

    if source_rel:
        try:
            lecture_root = lecture_root_from_source_relative(source_rel)
            rel_root = lecture_root.relative_to(APP_ROOT).as_posix()
            lecture_folder_name = lecture_root.name
            outputs_rel = f"{rel_root}/outputs"
            full_root = str(lecture_root.resolve())
        except (OSError, ValueError):
            pass

    if course_slug:
        full_course_dir = str((APP_ROOT / "courses" / course_slug).resolve())

    source_file_name = lecture.get("source_file_name") or ""
    if source_rel and "/" in source_rel:
        parent = source_rel.rsplit("/", 1)[0]
        source_dir_rel = f"{parent}/source"
    else:
        source_dir_rel = ""

    source_files: list[dict[str, Any]] = []
    multi_source = False
    if lecture_root is not None:
        m = source_manifest.load_manifest(lecture_root)
        if m and isinstance(m.get("files"), list):
            for i, ent in enumerate(m["files"]):
                if not isinstance(ent, dict):
                    continue
                source_files.append(
                    {
                        "name": str(ent.get("name") or ""),
                        "rel_path": str(ent.get("rel_path") or "").replace("\\", "/"),
                        "role": str(ent.get("role") or "other"),
                        "is_primary": i == 0,
                    }
                )
            multi_source = len(source_files) > 1
    if not source_files and source_rel:
        source_files = [
            {
                "name": source_file_name or Path(source_rel).name,
                "rel_path": source_rel,
                "role": "lecture",
                "is_primary": True,
            }
        ]

    return {
        "course_folder_rel": f"courses/{course_slug}" if course_slug else "",
        "lecture_folder_rel": rel_root,
        "lecture_folder_name": lecture_folder_name,
        "source_dir_rel": source_dir_rel,
        "source_file_rel": source_rel,
        "source_file_name": source_file_name,
        "outputs_rel": outputs_rel,
        "extracted_rel": extracted_rel,
        "full_lecture_root": full_root,
        "full_course_dir": full_course_dir,
        "source_files": source_files,
        "multi_source": multi_source,
    }


def attach_disk_folder_names(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same as :func:`enrich_lecture_rows_for_course_ui` (alias for home/search lists)."""
    return enrich_lecture_rows_for_course_ui(rows)
