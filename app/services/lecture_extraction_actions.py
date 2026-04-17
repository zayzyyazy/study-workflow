"""Re-run extraction and replace source file; shared extraction → DB → meta updates."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from app.config import APP_ROOT
from app.services import extraction_service, lecture_service, storage_service
from app.services import lecture_meta
from app.services import lecture_paths
from app.services import source_manifest
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


def _write_combined_and_sync(
    *,
    lecture_id: int,
    lecture_root: Path,
    files: list[dict],
    course_name: str,
    lecture_title: str,
    db_created_at: str,
) -> Tuple[str, Optional[str], str]:
    """Write extracted_text.txt from combined sources; update DB + meta."""
    ok, text, detail = source_manifest.combine_extracted_text(lecture_root, files)
    extracted_rel: Optional[str] = None
    msg = detail

    primary = files[0]
    primary_rel = str(primary["rel_path"]).replace("\\", "/")
    primary_name = str(primary["name"])

    if ok and text.strip():
        ext_path = lecture_root / "extracted_text.txt"
        ext_path.write_text(text, encoding="utf-8")
        extracted_rel = _relative(ext_path)
        status = READY_AFTER_EXTRACTION
        if len(files) > 1:
            msg = f"Combined text from {len(files)} source file(s). {detail}"
        elif not msg:
            msg = "Text extracted successfully."
    else:
        _remove_extracted_text(lecture_root)
        status = "extraction_failed"
        if not msg:
            msg = detail or "Extraction produced no text."

    display_name = primary_name
    if len(files) > 1:
        display_name = f"{primary_name} + {len(files) - 1} more"

    lecture_service.set_lecture_source_and_extraction(
        lecture_id,
        source_file_name=display_name,
        source_file_path=primary_rel,
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
        source_file_name=primary_name,
        source_rel_posix=primary_rel,
        extracted_rel_posix=extracted_rel,
        status=status,
        db_created_at=str(lec["created_at"]),
        extraction_message=msg,
        generated_artifacts=[],
        generation_message="",
        drop_lecture_analysis=True,
    )
    return status, extracted_rel, msg


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
    Persist manifest with a single primary source and run extraction (legacy-compatible).
    """
    safe_name = source_file.name
    files = [source_manifest.legacy_single_file_manifest(source_rel_posix, safe_name)["files"][0]]
    source_manifest.save_manifest(lecture_root, files)
    return _write_combined_and_sync(
        lecture_id=lecture_id,
        lecture_root=lecture_root,
        files=files,
        course_name=course_name,
        lecture_title=lecture_title,
        db_created_at=db_created_at,
    )


def re_run_extraction(lecture_id: int) -> Tuple[bool, str]:
    """Re-extract from all manifest sources (or legacy single file)."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    m = source_manifest.ensure_manifest(
        root,
        primary_rel_posix=str(lec["source_file_path"]).replace("\\", "/"),
        primary_name=str(lec["source_file_name"] or "source"),
    )
    files = list(m["files"])
    # Verify paths; drop missing
    files = [f for f in files if (APP_ROOT / str(f["rel_path"]).replace("\\", "/")).is_file()]
    if not files:
        return False, "No source files found on disk; add or replace sources."

    if source_manifest.load_manifest(root) is None:
        source_manifest.save_manifest(root, files)

    status, _ext, msg = _write_combined_and_sync(
        lecture_id=lecture_id,
        lecture_root=root,
        files=files,
        course_name=lec["course_name"],
        lecture_title=lec["title"],
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
    """Replace the primary source file; keep additional sources if present."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    source_dir = lecture_paths.source_dir_from_root(root)
    source_dir.mkdir(parents=True, exist_ok=True)

    m = source_manifest.ensure_manifest(
        root,
        primary_rel_posix=str(lec["source_file_path"]).replace("\\", "/"),
        primary_name=str(lec["source_file_name"] or "source"),
    )
    files = list(m["files"])

    safe_name = Path(original_filename).name
    old_primary_rel = str(files[0].get("rel_path", "")).replace("\\", "/")
    old_path = APP_ROOT / old_primary_rel if old_primary_rel else None
    if old_path and old_path.is_file():
        try:
            old_path.unlink()
        except OSError:
            pass

    dest = source_dir / safe_name
    storage_service.save_uploaded_file(file_obj, dest)
    new_rel = _relative(dest)

    files[0] = {"name": safe_name, "rel_path": new_rel, "role": "lecture"}
    source_manifest.save_manifest(root, files)

    status, _ext, msg = _write_combined_and_sync(
        lecture_id=lecture_id,
        lecture_root=root,
        files=files,
        course_name=lec["course_name"],
        lecture_title=lec["title"],
        db_created_at=str(lec["created_at"]),
    )

    lec2 = lecture_service.get_lecture_by_id(lecture_id)
    if not lec2:
        return False, "Lecture missing after update."
    if lec2["status"] == READY_AFTER_EXTRACTION:
        return True, msg or "Source updated and text extracted successfully."
    return True, msg or f"Source updated; extraction issue (status: {lec2['status']})."


def add_source_file(
    lecture_id: int,
    original_filename: str,
    file_obj: BinaryIO,
    role: Optional[str] = None,
) -> Tuple[bool, str]:
    """Add another source file under source/ and re-run combined extraction."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    root = lecture_paths.lecture_root_from_source_relative(lec["source_file_path"])
    source_dir = lecture_paths.source_dir_from_root(root)
    source_dir.mkdir(parents=True, exist_ok=True)

    m = source_manifest.ensure_manifest(
        root,
        primary_rel_posix=str(lec["source_file_path"]).replace("\\", "/"),
        primary_name=str(lec["source_file_name"] or "source").split(" + ")[0].strip(),
    )
    files = list(m["files"])

    safe_name = Path(original_filename).name
    dest = source_manifest.uniquify_dest(source_dir, safe_name)
    storage_service.save_uploaded_file(file_obj, dest)
    new_rel = _relative(dest)
    r = (role or "").strip().lower()
    if r in ("lecture", "exercise", "notes", "other"):
        mapped: source_manifest.SourceRole = r  # type: ignore[assignment]
    else:
        mapped = source_manifest.infer_role(safe_name)
    files.append({"name": dest.name, "rel_path": new_rel, "role": mapped})
    source_manifest.save_manifest(root, files)

    _status, _ext, msg = _write_combined_and_sync(
        lecture_id=lecture_id,
        lecture_root=root,
        files=files,
        course_name=lec["course_name"],
        lecture_title=lec["title"],
        db_created_at=str(lec["created_at"]),
    )
    lec2 = lecture_service.get_lecture_by_id(lecture_id)
    if lec2 and lec2["status"] == READY_AFTER_EXTRACTION:
        return True, msg or f"Added {dest.name}; combined extraction OK."
    return True, msg or "File added; check extraction status."
