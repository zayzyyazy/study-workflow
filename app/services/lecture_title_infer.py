"""Infer a human lecture title from extracted PDF/slide text (deterministic, no AI)."""

from __future__ import annotations

import re
from typing import Optional

# Head window: title slides usually appear early
_HEAD_CHARS = 14000
_MAX_LINE_LEN = 140
_MIN_TITLE_LEN = 8
# Strong preference for title-like lines in the first ~2 slides worth of text
_EARLY_LINE_CAP = 48


def _looks_like_noise(line: str) -> bool:
    s = line.strip()
    if len(s) < _MIN_TITLE_LEN:
        return True
    low = s.lower()
    noise = (
        "seite",
        "slide",
        "folie",
        "page",
        "moodle",
        "university",
        "universität",
        "copyright",
        "©",
        "http",
        "www.",
        "inhaltsverzeichnis",
        "department",
        "fachbereich",
        "vorwort",
        "presented by",
        "professor",
        "prof.",
    )
    if any(x in low for x in noise):
        return True
    if re.fullmatch(r"[\d\s.\-–—]+", s):
        return True
    if len(s) > _MAX_LINE_LEN:
        return True
    return False


def _clean_title_candidate(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[#•\-\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[: _MAX_LINE_LEN]


# "Lecture 2: Foo" / "Vorlesung 3 – Bar" / "Lecture 02 - Title" / "Week 4: Graphs"
_LECTURE_NUM = re.compile(
    r"^(?:lecture|lec|vorlesung|unit|kapitel|chapter|week|teil|part|session)\s*[:\-–]?\s*(\d{1,2})\s*[:\-–]\s*(.+)$",
    re.I,
)
# Same but number immediately after keyword: "Vorlesung 5 Graphs"
_LECTURE_NUM_TIGHT = re.compile(
    r"^(?:lecture|lec|vorlesung|unit|kapitel|chapter|week|teil|part|session)\s+(\d{1,2})\s+(.+)$",
    re.I,
)
_TITLE_COLON = re.compile(
    r"^(?:title|titel|topic|thema)\s*[:\-–]\s*(.+)$",
    re.I,
)


def _normalize_title_case(s: str) -> str:
    """Readable casing when slides export as ALL CAPS."""
    t = s.strip()
    if not t:
        return t
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 8:
        ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if ratio > 0.75 and len(t) <= 100:
            return t[:1] + t[1:].lower() if len(t) > 1 else t
    return t


def infer_base_title_from_extracted_text(head_text: str, *, fallback: str) -> str:
    """
    Best-effort title from the start of extracted text.
    Returns fallback if nothing reliable is found.
    """
    text = (head_text or "")[:_HEAD_CHARS]
    if not text.strip():
        return fallback

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # (score, title) — higher is better
    scored: list[tuple[float, str]] = []

    def add_candidate(raw: str, *, line_idx: int, base: float) -> None:
        c = _clean_title_candidate(raw)
        if not c or _looks_like_noise(c):
            return
        score = base + min(len(c), 80) * 0.15
        if line_idx < 12:
            score += 18.0
        elif line_idx < _EARLY_LINE_CAP:
            score += 10.0
        if re.search(r"(?:und|and|oder)\s+", c, re.I):
            score += 5.0
        # Prefer substantive topic words over single-word stubs
        words = c.split()
        if len(words) >= 3:
            score += 6.0
        scored.append((score, c))

    for line_idx, ln in enumerate(lines[:90]):
        m = re.match(r"^#{1,3}\s+(.+)$", ln)
        if m:
            add_candidate(m.group(1), line_idx=line_idx, base=14.0)
        m2 = _LECTURE_NUM.match(ln)
        if m2:
            add_candidate(m2.group(2), line_idx=line_idx, base=22.0)
        m3 = _LECTURE_NUM_TIGHT.match(ln)
        if m3:
            add_candidate(m3.group(2), line_idx=line_idx, base=20.0)
        tc = _TITLE_COLON.match(ln)
        if tc:
            add_candidate(tc.group(1), line_idx=line_idx, base=16.0)
        if len(ln) < 120 and not ln.startswith("#"):
            plain = _clean_title_candidate(re.sub(r"^\d+\.\s*", "", ln))
            if plain and len(plain) >= _MIN_TITLE_LEN:
                if re.search(r"[A-ZÄÖÜa-zäöü]", plain):
                    add_candidate(plain, line_idx=line_idx, base=4.0)

    if scored:
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        best = re.sub(
            r"^(?:vorlesung|lecture|lec|unit|week|teil|part|session)\s*\d{1,2}\s*[:\-–]\s*",
            "",
            best,
            flags=re.I,
        ).strip()
        best = re.sub(
            r"^(?:vorlesung|lecture|lec|unit)\s+\d{1,2}\s+",
            "",
            best,
            flags=re.I,
        ).strip()
        if len(best) >= _MIN_TITLE_LEN:
            best = _normalize_title_case(best[:100])
            return best

    return fallback
