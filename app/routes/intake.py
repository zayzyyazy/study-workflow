"""Smart PDF intake: preview inference, optional auto-commit, confirm fallback."""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT, DATA_DIR
from app.services import course_service, lecture_service, lecture_upload
from app.services import intake_temp_store, pdf_intake_inference
from app.services.mini_help_service import context_for_request

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()

_INTAKE_DIR = DATA_DIR / "intake_tmp"


@router.get("/pdf", response_class=RedirectResponse)
def redirect_legacy_pdf_route() -> RedirectResponse:
    """Older Study AI / bookmarks used ``/pdf``; intake lives at ``/intake``."""
    return RedirectResponse(url="/intake", status_code=307)


@router.get("/add-pdf", response_class=RedirectResponse)
def redirect_legacy_add_pdf_route() -> RedirectResponse:
    """Optional legacy alias for the intake page."""
    return RedirectResponse(url="/intake", status_code=307)


def _ensure_intake_dir() -> Path:
    _INTAKE_DIR.mkdir(parents=True, exist_ok=True)
    return _INTAKE_DIR


def _lecture_titles_by_course() -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for c in course_service.list_courses():
        cid = int(c["id"])
        rows = lecture_service.list_lectures_for_course(cid)
        out[cid] = [str(r["title"]) for r in rows[:24]]
    return out


@router.get("/intake", response_class=HTMLResponse)
def intake_page(request: Request, course_id: Optional[int] = None) -> HTMLResponse:
    courses = course_service.list_courses()
    default_course_id: int | None = course_id
    if default_course_id is not None:
        if not any(int(c["id"]) == default_course_id for c in courses):
            default_course_id = None
    return templates.TemplateResponse(
        request,
        "intake.html",
        {
            "title": "Add PDF",
            "courses": courses,
            "default_course_id": default_course_id,
            "mini_help_context": context_for_request(request, "intake"),
        },
    )


@router.post("/intake/preview")
async def intake_preview(file: UploadFile = File(...)) -> JSONResponse:
    name = (file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported in smart intake."}, status_code=400)
    dest_dir = _ensure_intake_dir()
    dest = dest_dir / f"{secrets.token_hex(14)}.pdf"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as e:
        return JSONResponse({"error": f"Could not save upload: {e}"}, status_code=500)

    courses = course_service.list_courses()
    titles_by = _lecture_titles_by_course()
    try:
        analysis = pdf_intake_inference.analyze_pdf_for_intake(
            dest,
            courses=courses,
            lecture_titles_by_course=titles_by,
            original_filename=name,
        )
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return JSONResponse({"error": f"Could not read PDF: {e}"}, status_code=400)

    token = intake_temp_store.store_file(dest, name)
    top = analysis.ranked_courses[0] if analysis.ranked_courses else None
    top_score = float(top["score"]) if top else 0.0
    auto_commit = bool(
        analysis.auto_place_course
        and analysis.auto_place_kind
        and top is not None
        and top_score > 0.0
    )
    payload: dict[str, Any] = {
        "token": token,
        "original_filename": name,
        "pdf_title": analysis.pdf_title,
        "material_kind": analysis.material_kind,
        "material_scores": analysis.material_scores,
        "material_confidence": analysis.material_confidence,
        "auto_place_kind": analysis.auto_place_kind,
        "material_note": analysis.material_note,
        "ranked_courses": analysis.ranked_courses[:10],
        "course_confidence": analysis.course_confidence,
        "course_margin": analysis.course_margin,
        "auto_place_course": analysis.auto_place_course,
        "auto_commit": auto_commit,
        "doc_tokens_sample": analysis.doc_tokens_sample,
    }
    return JSONResponse(payload)


@router.post("/intake/discard", response_model=None)
def intake_discard(token: str = Form(...)) -> RedirectResponse:
    ent = intake_temp_store.pop_entry((token or "").strip())
    if ent:
        path, _ = ent
        path.unlink(missing_ok=True)
    return RedirectResponse(url="/intake", status_code=303)


@router.post("/intake/commit", response_model=None)
async def intake_commit(
    request: Request,
    token: str = Form(...),
    course_id: str = Form(""),
    new_course_name: str = Form(""),
    material_kind: str = Form("lecture"),
    lecture_title: str = Form(""),
) -> RedirectResponse | HTMLResponse:
    tok = (token or "").strip()
    ent = intake_temp_store.pop_entry(tok)
    if not ent:
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": None,
                "error": "That upload expired. Drop the PDF again to continue.",
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=400,
        )

    path, original_filename = ent
    if not path.is_file():
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": None,
                "error": "That upload expired. Drop the PDF again to continue.",
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=400,
        )

    cid: Optional[int] = None
    new_name = (new_course_name or "").strip()
    if new_name:
        cid = None
    elif (course_id or "").strip():
        try:
            cid = int(course_id)
        except ValueError:
            cid = None
    if not new_name and cid is None:
        path.unlink(missing_ok=True)
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": None,
                "error": "Pick a course or enter a new course name.",
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=400,
        )

    orig = (original_filename or path.name).strip()
    if not orig.lower().endswith(".pdf"):
        orig = f"{orig}.pdf"

    lec: dict[str, Any] | None = None
    try:
        with path.open("rb") as fobj:
            lec = lecture_upload.create_lecture_from_upload(
                course_id=cid if not new_name else None,
                new_course_name=new_name if new_name else None,
                lecture_title=(lecture_title or "").strip(),
                original_filename=orig,
                file_obj=fobj,
                material_kind=material_kind or "lecture",
            )
    except ValueError as e:
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": cid,
                "error": str(e),
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=400,
        )
    except OSError as e:
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": cid,
                "error": f"Could not save file: {e}",
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=500,
        )
    finally:
        path.unlink(missing_ok=True)

    if lec is None:
        courses = course_service.list_courses()
        return templates.TemplateResponse(
            request,
            "intake.html",
            {
                "title": "Add PDF",
                "courses": courses,
                "default_course_id": cid,
                "error": "Upload did not complete.",
                "mini_help_context": context_for_request(request, "intake"),
            },
            status_code=500,
        )

    lid = int(lec["id"])
    return RedirectResponse(url=f"/lectures/{lid}", status_code=303)
