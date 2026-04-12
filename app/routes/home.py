"""Home page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_service, lecture_service
from app.services.dashboard_service import get_home_dashboard

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    courses = course_service.list_courses()
    recent = lecture_service.list_recent_lectures(limit=12)
    err = request.query_params.get("error")
    dash = get_home_dashboard()
    q = (request.query_params.get("q") or "").strip()
    search_hits = lecture_service.search_lectures_global(q) if q else []
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "title": "Home",
            "courses": courses,
            "recent_lectures": recent,
            "error": err,
            "course_count": dash["course_count"],
            "lecture_count": dash["lecture_count"],
            "status_counts": dash["status_counts"],
            "needs_attention": dash["needs_attention"],
            "search_q": q,
            "search_hits": search_hits,
        },
    )
