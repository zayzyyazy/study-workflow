"""Course detail."""

import io
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_ROOT
from app.services import course_service, lecture_service
from app.services.storage_view import enrich_lecture_rows_for_course_ui
from app.services.bulk_generation_service import run_bulk_generate_ready_in_course
from app.services.course_index_service import aggregate_course_concepts_filtered
from app.services.export_zip_service import zip_course_export
from app.services.lecture_service import KNOWN_LECTURE_STATUSES

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
    return templates.TemplateResponse(
        request,
        "course_detail.html",
        {
            "title": course["name"],
            "course": course,
            "lectures": lectures,
            "notice": notice,
            "error": err,
            "concept_rows": concept_rows,
            "lec_q": lec_q,
            "lec_status": lec_status,
            "concept_q": concept_q,
            "active_concept_id": only_concept_id,
            "lecture_statuses": KNOWN_LECTURE_STATUSES,
            "clear_concept_href": clear_concept_href,
            "clear_lecture_href": clear_lecture_href,
        },
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
