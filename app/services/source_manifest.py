"""
Multi-source files per lecture: manifest beside meta.json, combined extraction.

Legacy lectures: no manifest file — we treat DB primary path as the only source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from app.config import APP_ROOT
from app.services import extraction_service
from app.services import lecture_meta

MANIFEST_NAME = "source_manifest.json"

SourceRole = Literal["lecture", "exercise", "notes", "other"]


def manifest_path(lecture_root: Path) -> Path:
    return lecture_root / MANIFEST_NAME


def infer_role(filename: str) -> SourceRole:
    """Heuristic role from filename (user can override on upload)."""
    lower = filename.lower()
    if any(
        x in lower
        for x in (
            "uebung",
            "übung",
            "ubung",
            "exercise",
            "sheet",
            "blatt",
            "homework",
            "assignment",
            "aufgabe",
            "problem",
            "klausur",
            "exam",
            "quiz",
            "tutorium",
            "tutorial",
            "loesung",
            "lösung",
            "solution",
        )
    ):
        return "exercise"
    if any(x in lower for x in ("note", "notiz", "notizen", "handout", "skript")):
        return "notes"
    if any(x in lower for x in ("slide", "vorlesung", "lecture", "kapitel", "chapter")):
        return "lecture"
    return "other"


def load_manifest(lecture_root: Path) -> dict[str, Any] | None:
    p = manifest_path(lecture_root)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == 1 and isinstance(data.get("files"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def save_manifest(lecture_root: Path, files: list[dict[str, Any]]) -> None:
    lecture_root.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "files": files}
    manifest_path(lecture_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def legacy_single_file_manifest(source_rel_posix: str, filename: str, role: SourceRole = "lecture") -> dict[str, Any]:
    return {"version": 1, "files": [{"name": filename, "rel_path": source_rel_posix.replace("\\", "/"), "role": role}]}


def ensure_manifest(
    lecture_root: Path,
    *,
    primary_rel_posix: str,
    primary_name: str,
) -> dict[str, Any]:
    """Load manifest or create in-memory legacy view (does not write unless caller saves)."""
    loaded = load_manifest(lecture_root)
    if loaded:
        return loaded
    return legacy_single_file_manifest(primary_rel_posix, primary_name)


def uniquify_dest(source_dir: Path, desired: str) -> Path:
    """If file exists, append _2, _3, ... before extension."""
    p = source_dir / desired
    if not p.exists():
        return p
    stem = Path(desired).stem
    suf = Path(desired).suffix
    for n in range(2, 500):
        alt = source_dir / f"{stem}_{n}{suf}"
        if not alt.exists():
            return alt
    return source_dir / f"{stem}_copy{suf}"


def combine_extracted_text(lecture_root: Path, files: list[dict[str, Any]]) -> tuple[bool, str, str]:
    """
    Extract each file in order; concatenate with clear headers when multiple sources.
    Single file: raw text only (matches legacy single-upload behavior).
    Returns (ok, combined_text, human_message).
    """
    if len(files) == 1:
        entry = files[0]
        rel = (entry.get("rel_path") or "").replace("\\", "/")
        path = APP_ROOT / rel
        if not path.is_file():
            return False, "", f"Source file missing: {rel}"
        ex = extraction_service.extract_text_from_file(path)
        if ex.ok and ex.text.strip():
            return True, ex.text.strip(), ex.message or "Extracted text."
        return False, "", ex.message or "Extraction produced no text."

    parts: list[str] = []
    msgs: list[str] = []
    for entry in files:
        rel = (entry.get("rel_path") or "").replace("\\", "/")
        name = str(entry.get("name") or Path(rel).name)
        role = str(entry.get("role") or "other")
        path = APP_ROOT / rel
        if not path.is_file():
            msgs.append(f"missing:{name}")
            continue
        ex = extraction_service.extract_text_from_file(path)
        if ex.message:
            msgs.append(f"{name}: {ex.message[:80]}")
        if ex.ok and ex.text.strip():
            header = f"\n\n---\n\n## Source: {name}\n**Role:** {role}\n\n"
            parts.append(header + ex.text.strip())
        elif not ex.ok:
            parts.append(
                f"\n\n---\n\n## Source: {name}\n*(extraction failed: {ex.message or 'unknown'})*\n"
            )

    combined = "\n".join(parts).strip()
    if not combined:
        return False, "", "No extractable text from any source file."
    return True, combined, "; ".join(msgs[:5]) if msgs else "Combined extraction OK."


_SOURCE_HEADER = re.compile(
    r"^## Source:\s*(?P<name>[^\n]+)\n\*\*Role:\*\*\s*(?P<role>[a-zA-Z_]+)",
    re.MULTILINE,
)


def split_combined_extracted_text(full_text: str) -> tuple[str, str, str]:
    """
    Split multi-source extracted_text.txt into role buckets (body text only, no per-source headers).

    Returns (lecture_core, exercise_text, notes_text) where lecture_core merges roles
    lecture + notes + other. Legacy single-source text (no ``## Source:`` markers)
    is returned as (stripped full text, "", "").

    Raw extraction is unchanged on disk; this is for analysis and generation weighting only.
    """
    text = (full_text or "").strip()
    if not text:
        return "", "", ""
    if "## Source:" not in text or "**Role:**" not in text:
        return text, "", ""

    lecture_parts: list[str] = []
    exercise_parts: list[str] = []
    notes_parts: list[str] = []

    for chunk in re.split(r"\n\n---\n\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _SOURCE_HEADER.match(chunk)
        if not m:
            lecture_parts.append(chunk)
            continue
        role = m.group("role").lower().strip()
        body = chunk[m.end() :].strip()
        if role == "exercise":
            exercise_parts.append(body)
        elif role == "notes":
            notes_parts.append(body)
        else:
            lecture_parts.append(body)

    lecture_core = "\n\n".join(lecture_parts).strip()
    exercise_only = "\n\n".join(exercise_parts).strip()
    notes_only = "\n\n".join(notes_parts).strip()
    # Notes support the main teaching line — fold into lecture core for structure/topics.
    if notes_only:
        lecture_core = (lecture_core + "\n\n" + notes_only).strip() if lecture_core else notes_only

    if not lecture_core and (exercise_only or notes_only):
        # Degenerate manifest; fall back so callers still have material
        return text, exercise_only, ""

    return lecture_core, exercise_only, ""


def relative_to_app(path: Path) -> str:
    return lecture_meta.relative_to_app(path)
