"""Re-run extraction and replace source file; shared extraction → DB → meta updates."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from app.config import APP_ROOT
from app.services import extraction_service, lecture_service
from app.services import lecture_meta
from app.services import lecture_paths
from app.services.lecture_statuses import READY_AFTER_EXTRACTION


def _relative(p: Path) -> str:
    return lecture_meta.relative_to_app(p)


def _remove_extracted_text(lecture_root: Path) -> None:
    p = lecture_root / "extracted_text.txt"
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


def apply_extraction_from_source_file(
    *,
    lecture_id: int,
    lecture_root: Path,
    source_file: Path,
    course_name: str,
    lecture_title: str,
    source_rel_posix: str,
    db_created_at: str,
) -> Tuple[str, Optional[str], str]:
    """
    Runs extraction, updates extracted_text.txt, returns (status, extracted_rel_or_none, message).
    Never raises for extraction failures.
    """
    extraction = extraction_service.extract_text_from_file(source_file)
    msg = extraction.message or ""
    extracted_rel: Optional[str] = None

    if extraction.ok and extraction.text.strip():
        ext_path = lecture_root / "extracted_text.txt"
        ext_path.write_text(extraction.text, encoding="utf-8")
        extracted_rel = _relative(ext_path)
        status = READY_AFTER_EXTRACTION
        if not msg:
            msg = "Text extracted successfully."
    else:
        _remove_extracted_text(lecture_root)
        status = "extraction_failed"
        if not msg:
            msg = "Extraction produced no text."

    lecture_service.set_lecture_source_and_extraction(
        lecture_id,
        source_file_name=source_file.name,
        source_file_path=source_rel_posix,
        extracted_text_path=extracted_rel,
        status=status,
    )

    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return status, extracted_rel, msg

    lecture_meta.sync_meta_for_lecture(
        lecture_root,
        lecture_id=lecture_id,
        course_name=course_name,
        lecture_title=lecture_title,
        source_file_name=source_file.name,
        source_rel_posix=source_rel_posix,
        extracted_rel_posix=extracted_rel,
        status=status,
        db_created_at=str(lec["created_at"]),
        extraction_message=msg,
        generated_artifacts=[],
        generation_message="",
        drop_lecture_analysis=True,
    )
    return status, extracted_rel, msg


def re_run_extraction(lecture_id: int) -> Tuple[bool, str]:
    """Re-read source file and extract again. Returns (success, user_message)."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    src = APP_ROOT / lec["source_file_path"]
    if not src.is_file():
        return False, "Source file is missing on disk; try replacing the file."

    root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    status, _ext, msg = apply_extraction_from_source_file(
        lecture_id=lecture_id,
        lecture_root=root,
        source_file=src,
        course_name=lec["course_name"],
        lecture_title=lec["title"],
        source_rel_posix=lec["source_file_path"],
        db_created_at=str(lec["created_at"]),
    )
    if status == READY_AFTER_EXTRACTION:
        return True, msg
    return True, msg


def replace_source_file(
    lecture_id: int,
    original_filename: str,
    file_obj: BinaryIO,
) -> Tuple[bool, str]:
    """Replace file in source/, run extraction, update DB and meta."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    source_dir = lecture_paths.source_dir_from_root(root)
    source_dir.mkdir(parents=True, exist_ok=True)

    for existing in source_dir.iterdir():
        if existing.is_file():
            try:
                existing.unlink()
            except OSError:
                pass

    safe_name = Path(original_filename).name
    dest = source_dir / safe_name
    from app.services import storage_service

    storage_service.save_uploaded_file(file_obj, dest)
    source_rel = _relative(dest)

    apply_extraction_from_source_file(
        lecture_id=lecture_id,
        lecture_root=root,
        source_file=dest,
        course_name=lec["course_name"],
        lecture_title=lec["title"],
        source_rel_posix=source_rel,
        db_created_at=str(lec["created_at"]),
    )

    lec2 = lecture_service.get_lecture_by_id(lecture_id)
    if not lec2:
        return False, "Lecture missing after update."
    if lec2["status"] == READY_AFTER_EXTRACTION:
        return True, "Source file replaced and text extracted successfully."
    return True, f"Source file replaced; extraction did not produce usable text (status: {lec2['status']})."
