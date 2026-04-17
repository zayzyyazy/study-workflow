"""Primary and legacy output filenames for study materials."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# New structure (01-05). Values are fallbacks tried in order when primary is missing (old lectures).
LEGACY_FALLBACKS: dict[str, tuple[str, ...]] = {
    "quick_overview": ("01_quick_overview.md", "02_summary.md"),
    "topic_map": ("02_topic_map.md", "02_glossary.md", "01_glossary.md"),
    "core_learning": (
        "03_core_learning.md",
        "03_teach_me.md",
        "02_teach_me.md",
        "03_topic_explanations.md",
    ),
    "revision_sheet": (
        "04_revision_sheet.md",
        "05_revision_sheet.md",
        "05_connections.md",
        "02_summary.md",
    ),
    "study_pack": ("05_study_pack.md", "06_study_pack.md"),
}


def resolve_existing_output(outputs_dir: Path, artifact_type: str) -> tuple[Optional[Path], str]:
    """
    Return (path to first existing file, filename for display) or (None, primary name).
    """
    names = LEGACY_FALLBACKS.get(artifact_type)
    if not names:
        return None, ""
    for n in names:
        p = outputs_dir / n
        if p.is_file():
            return p, n
    return None, names[0]


def _strip_duplicate_heading(body: str, heading: str, *, also: tuple[str, ...] = ()) -> str:
    """
    Remove a top-level ## heading from body if it duplicates the section label
    we are about to write into the study pack.
    Only removes the very first ## line if it matches (case-insensitive).
    `also` lists alternate H2 titles from older generations (e.g. Core Learning → Topic Lessons).
    """
    lines = body.splitlines()
    if not lines:
        return body
    first = lines[0].strip()
    if not first.startswith("## "):
        return body
    inner = first[3:].strip().lower()
    targets = {heading.lower(), *(x.lower() for x in also)}
    if inner in targets:
        return "\n".join(lines[1:]).lstrip("\n")
    return body


def build_study_pack_markdown(outputs_dir: Path) -> str:
    """
    Concatenate primary study-pack sections into one Markdown file (no LLM).
    Supports new (01-04) and legacy output structures. Skips missing sections.
    Removes duplicate top-level section headings to avoid "## Quick Overview" twice.
    """
    # (file candidates, study-pack section title, optional legacy H2 titles to strip from file body)
    sections: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = [
        (("01_quick_overview.md", "02_summary.md"), "Quick Overview", ()),
        (
            ("02_topic_map.md", "02_glossary.md", "01_glossary.md"),
            "Topic Roadmap",
            ("Topic Map", "Themen-Roadmap", "Kurzes Inhaltsverzeichnis"),
        ),
        (
            ("03_core_learning.md", "03_teach_me.md", "02_teach_me.md", "03_topic_explanations.md"),
            "Topic Lessons",
            ("Core Learning", "Topic-Lektionen", "Topic Lessons"),
        ),
        (("04_revision_sheet.md", "05_revision_sheet.md"), "Revision Sheet", ()),
    ]
    chunks: list[str] = []
    for candidates, title, legacy_h2 in sections:
        p: Optional[Path] = None
        for fname in candidates:
            candidate = outputs_dir / fname
            if candidate.is_file():
                p = candidate
                break
        if p is None:
            continue
        body = p.read_text(encoding="utf-8", errors="replace").strip()
        if not body:
            continue
        body = _strip_duplicate_heading(body, title, also=legacy_h2)
        chunks.append(f"\n\n---\n\n## {title}\n\n{body}")
    text = (
        "# Study Pack\n\n"
        "*Single-file combination of all sections (rebuilt after each successful generation).*"
        + ("".join(chunks) if chunks else "\n\n*(No section files were available to combine.)*")
    )
    return text.strip() + "\n"
