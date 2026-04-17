"""Orchestrate saving an upload, extraction, meta.json, and DB row."""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Optional

from app.services import course_service, extraction_service, lecture_service, storage_service
from app.services import lecture_title_infer
from app.services import lecture_meta
from app.services import source_manifest
from app.services.slugs import sanitize_folder_name
from app.services.lecture_statuses import READY_AFTER_EXTRACTION


def _clean_title_candidate(raw: str) -> str:
    """
    Make a readable title from a user string or filename stem.
    Keeps words human-friendly without over-normalizing.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    parts = s.split(" ")
    # Drop noisy leading lecture/week markers so names like lecture_3_sets -> sets.
    while parts and re.fullmatch(r"(lecture|lec|vorlesung|week|wk)\.?", parts[0], flags=re.I):
        parts = parts[1:]
        while parts and re.fullmatch(r"\d{1,3}", parts[0]):
            parts = parts[1:]
    s = " ".join(parts).strip()
    if not s:
        return ""

    s = s.title()
    # Keep titles reasonably short and filesystem-safe.
    s = sanitize_folder_name(s, max_length=70)
    return s


def _derive_base_title(lecture_title: str, original_filename: str) -> str:
    custom = _clean_title_candidate(lecture_title)
    if custom:
        return custom
    stem = Path(original_filename or "").stem
    from_filename = _clean_title_candidate(stem)
    if from_filename:
        return from_filename
    return "Untitled Lecture"


def create_lecture_from_upload(
    *,
    course_id: Optional[int],
    new_course_name: Optional[str],
    lecture_title: str,
    original_filename: str,
    file_obj: BinaryIO,
) -> dict:
    """
    Creates or picks a course, writes files under courses/, runs extraction, inserts lecture.
    Returns the lecture dict including 'id' for redirect.
    """
    lecture_title = (lecture_title or "").strip()

    new_name = (new_course_name or "").strip()
    if new_name:
        course = course_service.create_course(new_name)
    elif course_id is not None:
        course = course_service.get_course_by_id(course_id)
        if not course:
            raise ValueError("Selected course was not found.")
    else:
        raise ValueError("Choose an existing course or enter a new course name.")

    cid = int(course["id"])
    idx = lecture_service.lecture_index_for_course(cid)
    base_title = _derive_base_title(lecture_title, original_filename)
    folder_name = storage_service.build_lecture_directory_name(idx, base_title)
    course_folder = str(course["slug"])

    lecture_root, source_dir, _outputs = storage_service.ensure_lecture_paths(
        course_folder, folder_name
    )

    safe_name = Path(original_filename).name
    dest_file = source_dir / safe_name
    storage_service.save_uploaded_file(file_obj, dest_file)

    extraction = extraction_service.extract_text_from_file(dest_file)
    extracted_rel: Optional[str] = None
    extraction_note = extraction.message or ""

    if extraction.ok and extraction.text.strip():
        ext_path = storage_service.write_extracted_text(lecture_root, extraction.text)
        extracted_rel = lecture_meta.relative_to_app(ext_path)
        status = READY_AFTER_EXTRACTION
        if not extraction_note:
            extraction_note = "Text extracted successfully."
    else:
        status = "extraction_failed"
        if not extraction_note:
            extraction_note = "Extraction produced no text."

    final_base = base_title
    if extraction.ok and extraction.text.strip():
        inferred = lecture_title_infer.infer_base_title_from_extracted_text(
            extraction.text,
            fallback=base_title,
        )
        if inferred and len(inferred.strip()) >= 6:
            final_base = inferred.strip()
    display_title = f"Lecture {idx:02d} - {final_base}"

    source_rel = lecture_meta.relative_to_app(dest_file)

    source_manifest.save_manifest(
        lecture_root,
        source_manifest.legacy_single_file_manifest(source_rel, safe_name)["files"],
    )

    lec = lecture_service.insert_lecture(
        course_id=cid,
        title=display_title,
        source_file_name=safe_name,
        source_file_path=source_rel,
        extracted_text_path=extracted_rel,
        status=status,
    )

    lecture_meta.sync_meta_for_lecture(
        lecture_root,
        lecture_id=int(lec["id"]),
        course_name=course["name"],
        lecture_title=display_title,
        source_file_name=safe_name,
        source_rel_posix=source_rel,
        extracted_rel_posix=extracted_rel,
        status=status,
        db_created_at=str(lec["created_at"]),
        extraction_message=extraction_note,
        generated_artifacts=[],
        generation_message="",
        drop_lecture_analysis=True,
    )
    lec["extraction_message"] = extraction_note
    return lec
