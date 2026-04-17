"""Rebuild combined study pack from existing section files (no LLM)."""

from __future__ import annotations

from typing import Any

from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.study_output_paths import build_study_pack_markdown


def rebuild_study_pack_file(lecture: dict[str, Any]) -> tuple[bool, str]:
    """
    Rewrite ``05_study_pack.md`` (or ``06_study_pack.md`` for legacy lectures) by
    concatenating section files. Does not call OpenAI or regenerate individual sections.
    """
    sp = lecture.get("source_file_path")
    if not sp:
        return False, "No source path for this lecture."
    try:
        root = lecture_root_from_source_relative(str(sp))
    except (OSError, ValueError) as e:
        return False, str(e)
    outputs = root / "outputs"
    if not outputs.is_dir():
        return False, "No outputs folder found. Generate study materials first."
    text = build_study_pack_markdown(outputs)
    # Use new filename; legacy lectures that had 06_ keep working via LEGACY_FALLBACKS
    out = outputs / "05_study_pack.md"
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as e:
        return False, str(e)
    return True, "Study pack file refreshed from section files (no AI)."
