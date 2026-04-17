"""Infer a human lecture title from extracted PDF/slide text (deterministic, no AI)."""

from __future__ import annotations

import re
from typing import Optional

# Head window: title slides usually appear early
_HEAD_CHARS = 12000
_MAX_LINE_LEN = 140
_MIN_TITLE_LEN = 8


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
        "agenda",
        "overview",
        "department",
        "fachbereich",
        "vorwort",
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


# "Lecture 2: Foo" / "Vorlesung 3 – Bar" / "Lecture 02 - Title"
_LECTURE_NUM = re.compile(
    r"^(?:lecture|lec|vorlesung|unit|kapitel|chapter)\s*[:\-–]?\s*(\d{1,2})\s*[:\-–]\s*(.+)$",
    re.I,
)


def infer_base_title_from_extracted_text(head_text: str, *, fallback: str) -> str:
    """
    Best-effort title from the start of extracted text.
    Returns fallback if nothing reliable is found.
    """
    text = (head_text or "")[:_HEAD_CHARS]
    if not text.strip():
        return fallback

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates: list[str] = []

    for ln in lines[:80]:
        # Markdown headings
        m = re.match(r"^#{1,3}\s+(.+)$", ln)
        if m:
            inner = _clean_title_candidate(m.group(1))
            if inner and not _looks_like_noise(inner):
                candidates.append(inner)
        m2 = _LECTURE_NUM.match(ln)
        if m2:
            rest = _clean_title_candidate(m2.group(2))
            if rest and not _looks_like_noise(rest):
                candidates.append(rest)
        # Plain bold-ish lines (common in slides)
        if len(ln) < 120 and not ln.startswith("#"):
            plain = _clean_title_candidate(re.sub(r"^\d+\.\s*", "", ln))
            if plain and not _looks_like_noise(plain) and len(plain) >= _MIN_TITLE_LEN:
                # Prefer lines that look like titles (capital letter or German noun)
                if re.search(r"[A-ZÄÖÜ]", plain) or len(plain) >= 20:
                    candidates.append(plain)

    if candidates:
        scored = []
        for c in candidates:
            score = len(c)
            if re.search(r"(?:und|and|oder)\s+", c, re.I):
                score += 8
            scored.append((score, c))
        scored.sort(reverse=True)
        best = scored[0][1]
        best = re.sub(
            r"^(?:vorlesung|lecture|lec|unit)\s*\d{1,2}\s*[:\-–]\s*",
            "",
            best,
            flags=re.I,
        ).strip()
        if len(best) >= _MIN_TITLE_LEN:
            return best[:100]

    return fallback
