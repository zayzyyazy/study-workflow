"""Study planner page — schedule editor + deterministic dashboard."""

from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_service, planner_schedule_service, planner_service

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


@router.get("/planner", response_class=HTMLResponse)
def planner_page(request: Request) -> HTMLResponse:
    dash = planner_service.build_planner_dashboard()
    courses = course_service.list_courses()
    notice = request.query_params.get("notice")
    err = request.query_params.get("error")
    return templates.TemplateResponse(
        request,
        "planner.html",
        {
            "title": "Planner",
            "dash": dash,
            "courses": courses,
            "notice": notice,
            "error": err,
            "weekday_names": planner_service.WEEKDAY_NAMES_FORM,
        },
    )


@router.post("/planner/schedule/add", response_model=None)
def post_add_schedule(
    title: str = Form(...),
    kind: str = Form(...),
    recurrence: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    weekday: str = Form(""),
    specific_date: str = Form(""),
    course_id: str = Form(""),
) -> RedirectResponse:
    cid: str | None = course_id.strip() or None
    wd: int | None = None
    if recurrence == "weekly" and weekday.strip() != "":
        try:
            wd = int(weekday)
        except ValueError:
            wd = None
    sd = specific_date.strip() or None
    ok, msg = planner_schedule_service.add_schedule_item(
        title=title,
        kind=kind,
        recurrence=recurrence,
        start_time=start_time,
        end_time=end_time,
        course_id=int(cid) if cid and cid.isdigit() else None,
        weekday=wd,
        specific_date=sd,
    )
    if not ok:
        return RedirectResponse(url=f"/planner?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/planner?notice={quote(msg)}", status_code=303)


@router.post("/planner/schedule/{item_id}/delete", response_model=None)
def post_delete_schedule(item_id: int) -> RedirectResponse:
    ok, msg = planner_schedule_service.delete_schedule_item(item_id)
    if not ok:
        return RedirectResponse(url=f"/planner?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/planner?notice={quote(msg)}", status_code=303)
