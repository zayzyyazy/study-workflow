"""Uni task CRUD endpoints (small dashboard layer)."""

from urllib.parse import quote

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from app.services import uni_task_service

router = APIRouter()


@router.post("/uni-tasks/add", response_model=None)
def post_add_uni_task(
    title: str = Form(...),
    due_date: str = Form(""),
) -> RedirectResponse:
    ok, msg = uni_task_service.create_task(title=title, due_date=due_date.strip() or None)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)


@router.post("/uni-tasks/{task_id}/done", response_model=None)
def post_done_uni_task(task_id: int) -> RedirectResponse:
    ok, msg = uni_task_service.set_done(task_id, True)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)


@router.post("/uni-tasks/{task_id}/undo", response_model=None)
def post_undo_uni_task(task_id: int) -> RedirectResponse:
    ok, msg = uni_task_service.set_done(task_id, False)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)


@router.post("/uni-tasks/{task_id}/delete", response_model=None)
def post_delete_uni_task(task_id: int) -> RedirectResponse:
    ok, msg = uni_task_service.delete_task(task_id)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)


@router.post("/uni-tasks/{task_id}/edit", response_model=None)
def post_edit_uni_task(
    task_id: int,
    title: str = Form(...),
    due_date: str = Form(""),
) -> RedirectResponse:
    ok, msg = uni_task_service.update_task(task_id=task_id, title=title, due_date=due_date.strip() or None)
    if not ok:
        return RedirectResponse(url=f"/?error={quote(msg)}", status_code=303)
    return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)
