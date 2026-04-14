"""Lecture upload form and handler."""

from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_service
from app.services import lecture_upload

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


@router.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request, error: Optional[str] = None) -> HTMLResponse:
    courses = course_service.list_courses()
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "title": "Upload lecture",
            "courses": courses,
            "error": error,
        },
    )


@router.post("/upload", response_model=None)
async def upload_post(
    request: Request,
    lecture_title: str = Form(""),
    course_id: Optional[str] = Form(None),
    new_course_name: str = Form(""),
    file: UploadFile = File(...),
) -> RedirectResponse | HTMLResponse:
    courses = course_service.list_courses()
    new_name = (new_course_name or "").strip()
    cid: Optional[int] = None
    if course_id and str(course_id).strip():
        try:
            cid = int(course_id)
        except ValueError:
            return templates.TemplateResponse(
                request,
                "upload.html",
                {
                    "title": "Upload lecture",
                    "courses": courses,
                    "error": "Invalid course selection.",
                },
                status_code=400,
            )

    if not new_name and cid is None:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "title": "Upload lecture",
                "courses": courses,
                "error": "Select an existing course or enter a new course name.",
            },
            status_code=400,
        )

    if not file.filename:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "title": "Upload lecture",
                "courses": courses,
                "error": "Please choose a file to upload.",
            },
            status_code=400,
        )

    try:
        lec = lecture_upload.create_lecture_from_upload(
            course_id=cid if not new_name else None,
            new_course_name=new_name if new_name else None,
            lecture_title=(lecture_title or "").strip(),
            original_filename=file.filename,
            file_obj=file.file,
        )
    except ValueError as e:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "title": "Upload lecture",
                "courses": courses,
                "error": str(e),
            },
            status_code=400,
        )
    except OSError as e:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "title": "Upload lecture",
                "courses": courses,
                "error": f"Could not save file: {e}",
            },
            status_code=500,
        )

    lid = int(lec["id"])
    return RedirectResponse(url=f"/lectures/{lid}", status_code=303)
