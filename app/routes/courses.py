"""Course detail."""

import io
import shutil
from html import escape
from urllib.parse import quote, urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_map_service, course_service, course_summaries_service, lecture_service
from app.services.course_delete import delete_course
from app.services.mini_help_service import context_for_request
from app.services.storage_view import enrich_lecture_rows_for_course_ui
from app.services.bulk_generation_service import run_bulk_generate_ready_in_course
from app.services.course_index_service import aggregate_course_concepts_filtered
from app.services.export_zip_service import zip_course_export
from app.services.lecture_service import KNOWN_LECTURE_STATUSES, STUDY_PROGRESS_STATES

templates = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))
router = APIRouter()


def _course_filter_url(course_id: int, **params: str | int | None) -> str:
    parts: dict[str, str] = {}
    for k, v in params.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts[k] = s
    if not parts:
        return f"/courses/{course_id}"
    return f"/courses/{course_id}?{urlencode(parts)}"


@router.get("/courses/{course_id}/map", response_class=HTMLResponse)
def course_map_and_path(request: Request, course_id: int) -> HTMLResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    ctx = course_map_service.build_course_map_and_path(course_id)
    total_lectures = lecture_service.count_lectures_for_course(course_id)
    study_done = lecture_service.count_study_progress_in_course(course_id, "done")
    return templates.TemplateResponse(
        request,
        "course_study_plan.html",
        {
            "title": f"{course['name']} · Map & path",
            "course": course,
            "plan": ctx,
            "total_lectures_in_course": total_lectures,
            "study_done_in_course": study_done,
            "mini_help_context": context_for_request(
                request,
                "course",
                course_id=course_id,
                course_name=course["name"],
                lecture_count=total_lectures,
                study_done_in_course=study_done,
            ),
        },
    )


@router.get("/courses/{course_id}", response_class=HTMLResponse)
def course_detail(request: Request, course_id: int) -> HTMLResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    lec_q = (request.query_params.get("lec_q") or "").strip()
    lec_status = (request.query_params.get("status") or "").strip()
    concept_q = (request.query_params.get("concept_q") or "").strip()
    concept_raw = request.query_params.get("concept")
    only_concept_id: int | None = None
    if concept_raw is not None and str(concept_raw).strip() != "":
        try:
            only_concept_id = int(concept_raw)
        except ValueError:
            only_concept_id = None

    lectures = enrich_lecture_rows_for_course_ui(
        lecture_service.list_lectures_for_course_filtered(
            course_id,
            title_query=lec_q,
            status=lec_status,
        )
    )
    concept_rows = aggregate_course_concepts_filtered(
        course_id,
        name_query=concept_q,
        only_concept_id=only_concept_id,
    )
    for row in concept_rows:
        row["filter_href"] = _course_filter_url(
            course_id,
            concept=row["concept_id"],
            lec_q=lec_q,
            status=lec_status,
            concept_q=concept_q,
        )
    clear_concept_href = _course_filter_url(
        course_id,
        lec_q=lec_q,
        status=lec_status,
        concept_q=concept_q,
    )
    clear_lecture_href = _course_filter_url(
        course_id,
        concept_q=concept_q,
        concept=only_concept_id,
    )
    notice = request.query_params.get("notice")
    err = request.query_params.get("error")
    total_lectures_in_course = lecture_service.count_lectures_for_course(course_id)
    study_done_in_course = lecture_service.count_study_progress_in_course(course_id, "done")
    summary_rows: list[dict] = []
    for row in course_summaries_service.list_summary_files(str(course["slug"])):
        summary_rows.append(
            {
                **row,
                "view_href": f"/courses/{course_id}/summaries/file?name={quote(row['name'])}",
            }
        )
    return templates.TemplateResponse(
        request,
        "course_detail.html",
        {
            "title": course["name"],
            "course": course,
            "lectures": lectures,
            "summary_files": summary_rows,
            "summaries_path_hint": f"courses/{course['slug']}/summaries/",
            "total_lectures_in_course": total_lectures_in_course,
            "study_done_in_course": study_done_in_course,
            "notice": notice,
            "error": err,
            "concept_rows": concept_rows,
            "lec_q": lec_q,
            "lec_status": lec_status,
            "concept_q": concept_q,
            "active_concept_id": only_concept_id,
            "lecture_statuses": KNOWN_LECTURE_STATUSES,
            "study_progress_states": STUDY_PROGRESS_STATES,
            "clear_concept_href": clear_concept_href,
            "clear_lecture_href": clear_lecture_href,
            "mini_help_context": context_for_request(
                request,
                "course",
                course_id=course_id,
                course_name=course["name"],
                lecture_count=total_lectures_in_course,
                study_done_in_course=study_done_in_course,
            ),
        },
    )


@router.post("/courses/{course_id}/summaries/upload", response_model=None)
async def post_upload_course_summary(
    course_id: int,
    file: UploadFile = File(...),
) -> RedirectResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    slug = str(course["slug"])
    safe = course_summaries_service.safe_summary_basename(file.filename or "")
    if not safe:
        return RedirectResponse(
            url=f"/courses/{course_id}?error="
            + quote("Use a .md, .txt, .markdown, or .pdf file with a simple filename (no folders)."),
            status_code=303,
        )
    dest_dir = course_summaries_service.ensure_summaries_dir(slug)
    dest = dest_dir / safe
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as e:
        return RedirectResponse(
            url=f"/courses/{course_id}?error=" + quote(f"Could not save summary: {e}"),
            status_code=303,
        )
    return RedirectResponse(
        url=f"/courses/{course_id}?notice=" + quote(f"Saved your summary «{safe}»."),
        status_code=303,
    )


@router.get("/courses/{course_id}/summaries/file", response_model=None)
def get_course_summary_file(
    course_id: int,
    name: str = "",
) -> FileResponse | HTMLResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    path = course_summaries_service.resolved_summary_path(str(course["slug"]), name)
    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    suf = path.suffix.lower()
    if suf in (".md", ".txt", ".markdown"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raise HTTPException(status_code=500, detail="Could not read file") from None
        body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(path.name)}</title>
<link rel="stylesheet" href="/static/style.css"></head>
<body class="container" style="max-width:52rem;margin:1rem auto;">
<p class="muted small-print"><a href="/courses/{course_id}">← Back to {escape(course['name'])}</a></p>
<h1 style="font-size:1.1rem;margin:0.5rem 0;">{escape(path.name)}</h1>
<pre style="white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--border);padding:1rem;border-radius:8px;font-size:0.88rem;">{escape(text)}</pre>
</body></html>"""
        return HTMLResponse(body)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.post("/courses/{course_id}/reset-study-progress", response_model=None)
def post_reset_course_study_progress(
    course_id: int,
    confirm: str | None = Form(default=None),
) -> RedirectResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if confirm != "1":
        return RedirectResponse(
            url=f"/courses/{course_id}?error="
            + quote("Check the box to confirm resetting study progress for this course."),
            status_code=303,
        )
    n = lecture_service.reset_study_progress_for_course(course_id)
    return RedirectResponse(
        url=f"/courses/{course_id}?notice="
        + quote(f"Study progress reset for {n} lecture(s) in this course."),
        status_code=303,
    )


@router.get("/courses/{course_id}/export.zip")
def download_course_export(course_id: int) -> StreamingResponse:
    try:
        data, fname = zip_course_export(course_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/courses/{course_id}/bulk-generate", response_model=None)
def post_bulk_generate(course_id: int) -> RedirectResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    result = run_bulk_generate_ready_in_course(course_id)
    if not result.get("ok"):
        return RedirectResponse(
            url=f"/courses/{course_id}?error={quote(result.get('error', 'Bulk generate failed.'))}",
            status_code=303,
        )
    r = int(result["ready"])
    s = int(result["skipped"])
    ok_n = int(result["succeeded"])
    fail_n = int(result["failed"])
    if r == 0:
        msg = f"No lectures were ready for generation ({s} other lecture(s) in this course skipped)."
    else:
        msg = (
            f"Bulk generate finished: {ok_n} succeeded, {fail_n} failed, "
            f"{s} skipped (not ready for generation)."
        )
    return RedirectResponse(url=f"/courses/{course_id}?notice={quote(msg)}", status_code=303)


@router.post("/courses/{course_id}/rename", response_model=None)
def post_rename_course(
    course_id: int,
    new_name: str = Form(...),
) -> RedirectResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    ok, msg = course_service.rename_course(course_id, new_name)
    if ok:
        return RedirectResponse(
            url=f"/courses/{course_id}?notice={quote(msg)}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/courses/{course_id}?error={quote(msg)}",
        status_code=303,
    )


@router.get("/courses/{course_id}/confirm-delete", response_class=HTMLResponse)
def get_confirm_delete_course(request: Request, course_id: int) -> HTMLResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    lecture_count = lecture_service.count_lectures_for_course(course_id)
    return templates.TemplateResponse(
        request,
        "course_delete_confirm.html",
        {
            "title": f"Delete {course['name']}",
            "course": course,
            "lecture_count": lecture_count,
        },
    )


@router.post("/courses/{course_id}/delete", response_model=None)
def post_delete_course(
    course_id: int,
    confirm: str | None = Form(default=None),
) -> RedirectResponse:
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if confirm != "1":
        return RedirectResponse(
            url=f"/courses/{course_id}?error={quote('Check the box to confirm deletion.')}",
            status_code=303,
        )
    ok, msg = delete_course(course_id)
    if ok:
        return RedirectResponse(
            url=f"/?notice={quote(msg)}",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/courses/{course_id}?error={quote(msg)}",
        status_code=303,
    )
