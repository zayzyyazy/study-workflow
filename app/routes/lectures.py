"""Lecture detail and lifecycle actions."""

import io
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import lecture_service
from app.services.concept_service import list_concepts_for_lecture
from app.services.lecture_delete import delete_lecture
from app.services.lecture_extraction_actions import re_run_extraction, replace_source_file
from app.services.export_zip_service import zip_lecture_export
from app.services.lecture_generation import run_study_materials_generation
from app.services.lecture_meta import read_meta
from app.services.lecture_outputs_view import load_generation_sections
from app.services.lecture_paths import lecture_root_from_source_relative

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()

PREVIEW_CHARS = 6000


def _lecture_redirect(lecture_id: int, notice: str | None = None, error: str | None = None) -> RedirectResponse:
    q: list[str] = []
    if notice:
        q.append(f"notice={quote(notice)}")
    if error:
        q.append(f"error={quote(error)}")
    suffix = ("?" + "&".join(q)) if q else ""
    return RedirectResponse(url=f"/lectures/{lecture_id}{suffix}", status_code=303)


@router.get("/lectures/{lecture_id}", response_class=HTMLResponse)
def lecture_detail(request: Request, lecture_id: int) -> HTMLResponse:
    lecture = lecture_service.get_lecture_by_id(lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    preview = ""
    extracted_path = lecture.get("extracted_text_path")
    if extracted_path:
        p = APP_ROOT / extracted_path
        if p.is_file():
            try:
                full = p.read_text(encoding="utf-8", errors="replace")
                preview = full if len(full) <= PREVIEW_CHARS else full[:PREVIEW_CHARS] + "\n\n… (truncated)"
            except OSError:
                preview = ""

    notice = request.query_params.get("notice")
    err = request.query_params.get("error")

    generation_sections = load_generation_sections(lecture)
    lecture_concepts = list_concepts_for_lecture(lecture_id)

    lecture_analysis = None
    try:
        root = lecture_root_from_source_relative(lecture["source_file_path"])
        meta = read_meta(root)
        la = meta.get("lecture_analysis")
        lecture_analysis = la if isinstance(la, dict) else None
    except (OSError, ValueError, KeyError):
        lecture_analysis = None

    return templates.TemplateResponse(
        request,
        "lecture_detail.html",
        {
            "title": lecture["title"],
            "lecture": lecture,
            "extracted_preview": preview,
            "notice": notice,
            "error": err,
            "generation_sections": generation_sections,
            "lecture_concepts": lecture_concepts,
            "lecture_analysis": lecture_analysis,
        },
    )


@router.post("/lectures/{lecture_id}/re-extract", response_model=None)
def post_re_extract(lecture_id: int) -> RedirectResponse:
    ok, msg = re_run_extraction(lecture_id)
    if not ok:
        return _lecture_redirect(lecture_id, error=msg)
    return _lecture_redirect(lecture_id, notice=msg)


@router.post("/lectures/{lecture_id}/replace-source", response_model=None)
async def post_replace_source(lecture_id: int, file: UploadFile = File(...)) -> RedirectResponse:
    if not file.filename:
        return _lecture_redirect(lecture_id, error="No file selected.")
    ok, msg = replace_source_file(lecture_id, file.filename, file.file)
    if not ok:
        return _lecture_redirect(lecture_id, error=msg)
    return _lecture_redirect(lecture_id, notice=msg)


@router.get("/lectures/{lecture_id}/confirm-delete", response_class=HTMLResponse)
def confirm_delete(request: Request, lecture_id: int) -> HTMLResponse:
    lecture = lecture_service.get_lecture_by_id(lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return templates.TemplateResponse(
        request,
        "lecture_delete_confirm.html",
        {
            "title": f"Delete {lecture['title']}",
            "lecture": lecture,
        },
    )


@router.post("/lectures/{lecture_id}/delete", response_model=None)
def post_delete_lecture(lecture_id: int) -> RedirectResponse:
    ok, msg, course_id = delete_lecture(lecture_id)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    if course_id is None:
        return RedirectResponse(url="/", status_code=303)
    n = quote(msg)
    return RedirectResponse(url=f"/courses/{course_id}?notice={n}", status_code=303)


@router.post("/lectures/{lecture_id}/generate", response_model=None)
def post_generate(lecture_id: int) -> RedirectResponse:
    ok, msg = run_study_materials_generation(lecture_id)
    if not ok:
        return _lecture_redirect(lecture_id, error=msg)
    return _lecture_redirect(lecture_id, notice=msg)


@router.get("/lectures/{lecture_id}/export.zip")
def download_lecture_export(lecture_id: int) -> StreamingResponse:
    try:
        data, fname = zip_lecture_export(lecture_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
