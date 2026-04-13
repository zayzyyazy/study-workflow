"""Primary and legacy output filenames for study materials (study pack redesign)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# New study-pack files (primary). Values are fallbacks tried in order if primary missing (old lectures).
LEGACY_FALLBACKS: dict[str, tuple[str, ...]] = {
    "glossary": ("01_glossary.md",),
    "teach_me": ("02_teach_me.md", "02_summary.md", "03_topic_explanations.md"),
    "worked_examples": ("03_worked_examples.md", "04_deep_dive.md"),
    "mistakes_and_checks": ("04_mistakes_and_checks.md", "05_connections.md"),
    "revision_sheet": ("05_revision_sheet.md", "02_summary.md"),
    "study_pack": ("06_study_pack.md",),
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


def build_study_pack_markdown(outputs_dir: Path) -> str:
    """
    Concatenate primary study-pack sections into one Markdown file (no LLM).
    Skips missing sections.
    """
    sections: list[tuple[str, str]] = [
        ("01_glossary.md", "Glossary"),
        ("02_teach_me.md", "Teach Me"),
        ("03_worked_examples.md", "Worked Examples"),
        ("04_mistakes_and_checks.md", "Mistakes and Checks"),
        ("05_revision_sheet.md", "Revision Sheet"),
    ]
    chunks: list[str] = []
    for fname, title in sections:
        p = outputs_dir / fname
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8", errors="replace").strip()
        if not body:
            continue
        chunks.append(f"\n\n---\n\n## {title}\n\n{body}")
    text = (
        "# Study pack\n\n"
        "*Single-file combination of the sections below (generated after each successful run).*"
        + ("".join(chunks) if chunks else "\n\n*(No section files were available to combine.)*")
    )
    return text.strip() + "\n"
