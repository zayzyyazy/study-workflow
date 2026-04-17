"""Load generated Markdown for the lecture detail page."""

from __future__ import annotations

from typing import Any, Optional

from app.services.artifact_service import GENERATION_ARTIFACT_TYPES
from app.services.markdown_math import markdown_to_lecture_html
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.study_output_paths import resolve_existing_output

SECTION_TITLES = {
    "quick_overview": "Quick Overview",
    "topic_map": "Topic Roadmap",
    "core_learning": "Topic Lessons",
    "revision_sheet": "Revision Sheet",
    "study_pack": "Study pack (single file)",
}


def load_generation_sections(lecture: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One entry per expected output; includes HTML when a file exists.
    Tries new filenames first, then legacy filenames for older lectures.
    """
    root = lecture_root_from_source_relative(lecture["source_file_path"])
    outputs = root / "outputs"
    out: list[dict[str, Any]] = []
    for artifact_type, _primary in GENERATION_ARTIFACT_TYPES:
        path, fname_shown = resolve_existing_output(outputs, artifact_type)
        md: Optional[str] = None
        html = ""
        if path is not None and path.is_file():
            try:
                md = path.read_text(encoding="utf-8", errors="replace")
                html = markdown_to_lecture_html(md)
            except OSError:
                md = None
        out.append(
            {
                "artifact_type": artifact_type,
                "filename": fname_shown,
                "title": SECTION_TITLES.get(artifact_type, artifact_type),
                "markdown": md,
                "html": html,
            }
        )
    return out
