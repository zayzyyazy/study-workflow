"""Load generated Markdown for the lecture detail page."""

from __future__ import annotations

from typing import Any, Optional

import markdown

from app.services.artifact_service import GENERATION_ARTIFACT_TYPES
from app.services.lecture_paths import lecture_root_from_source_relative

SECTION_TITLES = {
    "glossary": "Glossary",
    "summary": "Summary",
    "topic_explanations": "Topic explanations",
    "deep_dive": "Deep dive",
    "connections": "Connections",
}


def load_generation_sections(lecture: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One entry per expected output file; includes HTML when the file exists.
    """
    root = lecture_root_from_source_relative(lecture["source_file_path"])
    outputs = root / "outputs"
    out: list[dict[str, Any]] = []
    for artifact_type, filename in GENERATION_ARTIFACT_TYPES:
        path = outputs / filename
        md: Optional[str] = None
        html = ""
        if path.is_file():
            try:
                md = path.read_text(encoding="utf-8", errors="replace")
                html = markdown.markdown(
                    md,
                    extensions=["fenced_code", "tables"],
                )
            except OSError:
                md = None
        out.append(
            {
                "artifact_type": artifact_type,
                "filename": filename,
                "title": SECTION_TITLES.get(artifact_type, artifact_type),
                "markdown": md,
                "html": html,
            }
        )
    return out
