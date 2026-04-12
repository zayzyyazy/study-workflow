"""
Extract key concept strings from generated Markdown files (deterministic, no LLM).

Uses glossary (primary), plus headings and bold terms from summary / topics / deep dive.
Skips connections — intra-lecture narrative, weaker for vocabulary terms.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.services.concept_normalize import clean_display_name, normalize_concept_key

# Source files (same names as generation pipeline)
SOURCES = (
    "01_glossary.md",
    "02_summary.md",
    "03_topic_explanations.md",
    "04_deep_dive.md",
)

MAX_CONCEPTS = 80
MIN_LEN = 2
MAX_LEN = 120

# Skip obvious non-concepts
_STOP = frozenset(
    {
        "introduction",
        "summary",
        "conclusion",
        "overview",
        "example",
        "examples",
        "note",
        "notes",
        "figure",
        "table",
        "lecture",
        "topic",
        "topics",
        "section",
        "glossary",
        "deep dive",
        "topic explanations",
        "connections",
        # German section titles from language-aware generation
        "glossar",
        "zusammenfassung",
        "vertiefung",
        "zusammenhänge",
        "themen und kurzerklärungen",
        "themen",
    }
)


def _skip_term(s: str) -> bool:
    t = s.strip()
    if len(t) < MIN_LEN or len(t) > MAX_LEN:
        return True
    if normalize_concept_key(t) in _STOP:
        return True
    if t.isdigit():
        return True
    return False


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
    if _skip_term(raw):
        return None
    return clean_display_name(raw)


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
    if _skip_term(cell):
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
        if not _skip_term(title):
            out.append(title)
    return out


def _parse_bold(text: str, limit: int = 35) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\*\*([^*]{2,80})\*\*", text):
        t = clean_display_name(m.group(1).strip())
        if not _skip_term(t):
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

    def add_many(items: list[str]) -> None:
        for item in items:
            disp = clean_display_name(item)
            if _skip_term(disp):
                continue
            key = normalize_concept_key(disp)
            if not key or key in _STOP:
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
            add_many(_parse_glossary(text))
        add_many(_parse_headings(text))
        if "summary" in fname or "deep" in fname or "topic" in fname:
            add_many(_parse_bold(text))

    return order[:MAX_CONCEPTS]
