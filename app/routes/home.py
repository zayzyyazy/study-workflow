"""Home page."""

from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_service, home_dashboard_service, lecture_service
from app.services.mini_help_service import context_for_request
from app.services.storage_view import attach_disk_folder_names

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    courses = course_service.list_courses_for_home_dashboard()
    err = request.query_params.get("error")
    notice = request.query_params.get("notice")
    study_totals = lecture_service.study_progress_library_totals()
    starred = attach_disk_folder_names(lecture_service.list_starred_lectures(limit=24))
    home_dash = home_dashboard_service.build_home_dashboard()
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "title": "Home",
            "courses": courses,
            "error": err,
            "notice": notice,
            "study_totals": study_totals,
            "starred_lectures": starred,
            "home_dash": home_dash,
            "mini_help_context": context_for_request(
                request,
                "home",
                study_done=study_totals.get("done"),
                study_total=study_totals.get("total"),
            ),
        },
    )


@router.post("/reset-study-progress", response_model=None)
def post_reset_study_progress(confirm: str | None = Form(default=None)) -> RedirectResponse:
    if confirm != "1":
        return RedirectResponse(
            url="/?error=" + quote("Check the box to confirm resetting study progress."),
            status_code=303,
        )
    n = lecture_service.reset_all_study_progress()
    return RedirectResponse(
        url="/?notice="
        + quote(f"Study progress reset for {n} lecture(s). Everything is Not started again."),
        status_code=303,
    )
