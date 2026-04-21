"""Degree / credit-point progress (local CP tracker)."""

from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import degree_progress_service
from app.services.mini_help_service import context_for_request

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


@router.get("/degree-progress", response_class=HTMLResponse)
def degree_progress_page(request: Request) -> HTMLResponse:
    summary = degree_progress_service.summarize()
    grouped = degree_progress_service.entries_by_category()
    notice = request.query_params.get("notice")
    err = request.query_params.get("error")
    return templates.TemplateResponse(
        request,
        "degree_progress.html",
        {
            "title": "Degree progress",
            "summary": summary,
            "grouped_entries": grouped,
            "target_cp": summary["target_cp"],
            "notice": notice,
            "error": err,
            "mini_help_context": context_for_request(request, "degree_progress"),
        },
    )


@router.post("/degree-progress/target", response_model=None)
def post_target_cp(target_cp: str = Form(...)) -> RedirectResponse:
    try:
        v = float((target_cp or "").replace(",", ".").strip())
    except ValueError:
        return RedirectResponse(
            url="/degree-progress?error=" + quote("Target CP must be a number."),
            status_code=303,
        )
    ok, msg = degree_progress_service.set_target_cp(v)
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)


@router.post("/degree-progress/add", response_model=None)
def post_add_entry(
    title: str = Form(...),
    cp: str = Form(...),
    category: str = Form(""),
) -> RedirectResponse:
    try:
        cp_val = float((cp or "").replace(",", ".").strip())
    except ValueError:
        return RedirectResponse(
            url="/degree-progress?error=" + quote("CP must be a number."),
            status_code=303,
        )
    ok, msg = degree_progress_service.add_entry(title=title, cp=cp_val, category=category.strip() or None)
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)


@router.post("/degree-progress/{entry_id}/done", response_model=None)
def post_mark_done(entry_id: int) -> RedirectResponse:
    ok, msg = degree_progress_service.set_done(entry_id, True)
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)


@router.post("/degree-progress/{entry_id}/open", response_model=None)
def post_mark_open(entry_id: int) -> RedirectResponse:
    ok, msg = degree_progress_service.set_done(entry_id, False)
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)


@router.post("/degree-progress/{entry_id}/delete", response_model=None)
def post_delete_entry(entry_id: int) -> RedirectResponse:
    ok, msg = degree_progress_service.delete_entry(entry_id)
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)


@router.post("/degree-progress/{entry_id}/edit", response_model=None)
def post_edit_entry(
    entry_id: int,
    title: str = Form(...),
    cp: str = Form(...),
    category: str = Form(""),
) -> RedirectResponse:
    try:
        cp_val = float((cp or "").replace(",", ".").strip())
    except ValueError:
        return RedirectResponse(
            url="/degree-progress?error=" + quote("CP must be a number."),
            status_code=303,
        )
    ok, msg = degree_progress_service.update_entry(
        entry_id, title=title, cp=cp_val, category=category.strip() or None
    )
    if not ok:
        return RedirectResponse(url="/degree-progress?error=" + quote(msg), status_code=303)
    return RedirectResponse(url="/degree-progress?notice=" + quote(msg), status_code=303)
