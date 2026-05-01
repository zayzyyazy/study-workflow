"""Orchestrate saving an upload, extraction, meta.json, and DB row."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

from app.services import course_service, extraction_service, lecture_service, storage_service
from app.services import lecture_title_infer
from app.services import lecture_meta
from app.services import source_manifest
from app.services import upload_title_cleanup
from app.services.pdf_intake_inference import read_pdf_metadata_title
from app.services.slugs import sanitize_folder_name
from app.services.lecture_statuses import READY_AFTER_EXTRACTION

_DISPLAY_PREFIX = {"lecture": "Lecture", "exercise": "Sheet", "material": "Material"}


def _normalize_material_kind(raw: str) -> str:
    k = (raw or "lecture").strip().lower() or "lecture"
    if k in ("sheet", "aufgabe", "aufgabenblatt", "ubung", "übung"):
        return "exercise"
    if k in ("lecture", "exercise", "material"):
        return k
    return "lecture"


def _finalize_readable_title(s: str, *, material_kind: str = "lecture") -> str:
    """Polish casing, strip unsafe chars, card-friendly length for folder + display."""
    raw = (s or "").strip()
    t = lecture_title_infer.squeeze_card_base(raw, material_kind=material_kind)
    if not t:
        t = lecture_title_infer._safe_short_fallback(material_kind)
    t = upload_title_cleanup.polish_readable_base(
        t, max_len=lecture_title_infer._CARD_BASE_MAX_CHARS + 2
    )
    t = lecture_title_infer._normalize_title_case(t)
    return sanitize_folder_name(t, max_length=52)


def _derive_base_title(
    lecture_title: str,
    original_filename: str,
    *,
    material_kind: str,
    pdf_metadata_title: str = "",
    course_name: str = "",
) -> str:
    """
    Human-readable base segment (no Lecture/Sheet prefix — that is added later with index).
    Uses optional PDF /Title metadata when it beats a scrubbed filename.
    """
    lt = (lecture_title or "").strip()
    if lt:
        seed = lt
    else:
        stem = Path(original_filename or "").stem
        stem_s = upload_title_cleanup.scrub_filename_stem(stem)
        meta_raw = (pdf_metadata_title or "").strip()
        meta_s = upload_title_cleanup.scrub_filename_stem(meta_raw) if meta_raw else ""
        seed = upload_title_cleanup.prefer_metadata_or_stem(meta_s, stem_s) if meta_s else stem_s
    if not (seed or "").strip():
        seed = "Untitled"
    seed = upload_title_cleanup.strip_redundant_material_prefix(seed.strip(), material_kind)
    seed = upload_title_cleanup.strip_duplicate_course_title(seed, course_name)
    out = _finalize_readable_title(seed, material_kind=material_kind)
    return out or "Untitled Lecture"


def create_lecture_from_upload(
    *,
    course_id: Optional[int],
    new_course_name: Optional[str],
    lecture_title: str,
    original_filename: str,
    file_obj: BinaryIO,
    material_kind: str = "lecture",
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
    mk = _normalize_material_kind(material_kind)
    idx = lecture_service.next_slot_index_for_material_kind(cid, mk)

    file_obj.seek(0)
    pdf_meta = ""
    tdir = Path(tempfile.mkdtemp(prefix="pdfmeta-"))
    try:
        tpath = tdir / "in.pdf"
        with tpath.open("wb") as out:
            shutil.copyfileobj(file_obj, out)
        if tpath.is_file() and tpath.stat().st_size > 0:
            pdf_meta = read_pdf_metadata_title(tpath)
    except OSError:
        pdf_meta = ""
    finally:
        shutil.rmtree(tdir, ignore_errors=True)
    file_obj.seek(0)

    base_title = _derive_base_title(
        lecture_title,
        original_filename,
        material_kind=mk,
        pdf_metadata_title=pdf_meta,
        course_name=str(course.get("name") or ""),
    )
    folder_name = storage_service.build_lecture_directory_name(idx, base_title, material_kind=mk)
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
            material_kind=mk,
        )
        if inferred and len(inferred.strip()) >= 4:
            if lecture_title_infer.is_unacceptable_card_title(inferred.strip(), mk):
                final_base = base_title
            else:
                final_base = inferred.strip()
    final_base = upload_title_cleanup.strip_redundant_material_prefix(final_base.strip(), mk)
    final_base = upload_title_cleanup.contextualize_upload_title(
        final_base,
        course_name=str(course.get("name") or ""),
        course_slug=str(course.get("slug") or ""),
        material_kind=mk,
    )
    final_base = _finalize_readable_title(final_base, material_kind=mk) or base_title
    prefix = _DISPLAY_PREFIX.get(mk, "Lecture")
    display_title = f"{prefix} {idx:02d} - {final_base}"

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
        material_kind=mk,
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
        material_kind=mk,
    )
    lec["extraction_message"] = extraction_note
    return lec
