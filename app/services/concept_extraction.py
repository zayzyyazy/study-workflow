"""
Extract key concept strings from generated Markdown files (deterministic, no LLM).

Uses glossary (primary), plus headings and bold terms from teach-me, worked examples, etc.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.services.concept_normalize import clean_display_name, normalize_concept_key
from app.services.concept_quality import is_noise_concept

# Must match primary output filenames (study pack is aggregate — not listed here)
SOURCES = (
    "01_quick_overview.md",
    "02_glossary.md",
    "03_teach_me.md",
    "04_examples_and_solutions.md",
    "05_revision_sheet.md",
)

MAX_CONCEPTS = 55
MAX_BOLD_PICKS = 22


def _from_glossary_line(line: str) -> str | None:
    raw = line.strip()
    if not raw.startswith(("-", "*", "•")) and not re.match(r"^\d+\.\s", raw):
        return None
    raw = re.sub(r"^\s*[-*•]\s+", "", raw)
    raw = re.sub(r"^\d+\.\s+", "", raw)
    raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
    for sep in (" — ", " – ", " - ", ": ", "："):
        if sep in raw:
            raw = raw.split(sep)[0]
            break
    raw = raw.strip().strip("*`").strip()
    raw = clean_display_name(raw)
    if is_noise_concept(raw, mode="glossary"):
        return None
    return raw


def _from_table_row(line: str) -> str | None:
    if "|" not in line or line.strip().startswith("|---"):
        return None
    parts = [p.strip() for p in line.split("|")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    cell = parts[0]
    if re.match(r"^-+$", cell.replace(" ", "")):
        return None
    if cell.lower() in ("term", "concept", "keyword", "name"):
        return None
    cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)
    cell = clean_display_name(cell)
    if is_noise_concept(cell, mode="glossary"):
        return None
    return cell


def _parse_glossary(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line and line.count("|") >= 2:
            t = _from_table_row(line)
            if t:
                out.append(t)
        else:
            t = _from_glossary_line(line)
            if t:
                out.append(t)
    return out


def _parse_headings(text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"^#{2,3}\s+(.+?)\s*$", text, re.MULTILINE):
        title = m.group(1).strip()
        title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
        title = clean_display_name(title)
        if not is_noise_concept(title, mode="strict"):
            out.append(title)
    return out


def _parse_bold(text: str, limit: int = 35) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\*\*([^*]{2,80})\*\*", text):
        t = clean_display_name(m.group(1).strip())
        if not is_noise_concept(t, mode="strict"):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def extract_concepts_from_outputs(outputs_dir: Path) -> list[str]:
    """
    Read generation outputs and return unique display names (stable order, capped).
    Deduplicates by normalized key; prefers first occurrence.
    """
    seen: dict[str, str] = {}
    order: list[str] = []

    def add_many(items: list[str], *, mode: str) -> None:
        for item in items:
            disp = clean_display_name(item)
            if is_noise_concept(disp, mode=mode):
                continue
            key = normalize_concept_key(disp)
            if not key:
                continue
            if key not in seen:
                seen[key] = disp
                order.append(disp)

    for fname in SOURCES:
        path = outputs_dir / fname
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "glossary" in fname.lower():
            add_many(_parse_glossary(text), mode="glossary")
        add_many(_parse_headings(text), mode="strict")
        if "02_glossary" not in fname and "01_glossary" not in fname:
            add_many(_parse_bold(text), mode="strict")

    return order[:MAX_CONCEPTS]
