"""Orchestrate saving an upload, extraction, meta.json, and DB row."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional

from app.services import course_service, extraction_service, lecture_service, storage_service
from app.services import lecture_meta
from app.services.lecture_statuses import READY_AFTER_EXTRACTION


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
    lecture_title = lecture_title.strip()
    if not lecture_title:
        raise ValueError("Lecture title is required.")

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
    folder_name = storage_service.build_lecture_directory_name(idx, lecture_title)
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

    source_rel = lecture_meta.relative_to_app(dest_file)

    lec = lecture_service.insert_lecture(
        course_id=cid,
        title=lecture_title,
        source_file_name=safe_name,
        source_file_path=source_rel,
        extracted_text_path=extracted_rel,
        status=status,
    )

    lecture_meta.sync_meta_for_lecture(
        lecture_root,
        lecture_id=int(lec["id"]),
        course_name=course["name"],
        lecture_title=lecture_title,
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
