"""Deterministic rules for concept quality: filtering noise and normalizing text."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Headings / bold: require a bit more substance than glossary lines.
_MIN_LEN_STRICT: Final = 3
_MIN_LEN_GLOSSARY: Final = 2
_MAX_LEN: Final = 120

_STOP: Final = frozenset(
    {
        "introduction",
        "intro",
        "overview",
        "background",
        "motivation",
        "summary",
        "conclusion",
        "discussion",
        "notation",
        "preliminaries",
        "preliminary",
        "definition",
        "definitions",
        "example",
        "examples",
        "remark",
        "remarks",
        "note",
        "notes",
        "figure",
        "figures",
        "table",
        "tables",
        "lecture",
        "topic",
        "topics",
        "section",
        "sections",
        "chapter",
        "part",
        "appendix",
        "glossary",
        "deep dive",
        "topic explanations",
        "connections",
        "outline",
        "agenda",
        "recap",
        "exercise",
        "exercises",
        "problem",
        "problems",
        "homework",
        "references",
        "bibliography",
        "abstract",
        "methods",
        "method",
        "results",
        "proof",
        "proofs",
        "lemma",
        "theorem",
        "corollary",
        "proposition",
        "glossar",
        "zusammenfassung",
        "vertiefung",
        "zusammenhänge",
        "themen und kurzerklärungen",
        "themen",
        "inhalt",
        "überblick",
        "checkliste",
        "checklist",
        "core learning",
        "quick overview",
        "topic map",
        "revision sheet",
        "study pack",
        "merkblatt",
        "klausurvorbereitung",
        "tiefe",
        "verbindungen",
        "depth",
        "connections",
        "priorität",
        "priority",
        "hinweis",
    }
)

# Multi-word / UI boilerplate from generated study materials (substring match, lowercased key)
_BOILERPLATE_SUBSTRINGS: Final = frozenset(
    {
        "core learning",
        "quick overview",
        "topic map",
        "revision sheet",
        "study pack",
        "inhalt der einheit",
        "inhalt dieser einheit",
        "checkliste",
        "check list",
        "topic lessons",
        "topic-lektionen",
        "inhaltsverzeichnis",
        "topic roadmap",
        "themen-roadmap",
        "organisatorisches",
        "übungsgruppe",
        "übungsgruppen",
        "übungsgruppenwahl",
        "tauschbörse",
        "nächste schritte",
        "moodle",
        "übungsablauf",
        "aufgabe zu",
        "auswendig lernen",
        "konzeptuell verstehen",
        "typische fragestellungen",
        "typische fehler",
        "typische fehltritte",
        "typische missverständnisse",
        "kurs-link",
        "warum wichtig",
        "was es ist",
        "prüfungsnähe",
        "prüfungsbezug",
        "übungs- und prüfungsbezug",
        "course link",
        "why it matters",
        "what it is",
        "typical misunderstandings",
        "typical slips",
        "tasks / exam angle",
        "aufgaben / prüfungsnähe",
        "**priorität:**",
        "priorität:",
    }
)


_ROMAN_NUM = re.compile(r"^[ivxlcdm]{1,8}$", re.I)


def strip_leading_numbering(text: str) -> str:
    """
    Remove common outline prefixes: ``1.``, ``2.3.``, ``(a)``, Roman ``IV.`` at the start.
    Applied repeatedly so nested numbering is peeled.
    """
    t = text.strip()
    if not t:
        return t
    for _ in range(5):
        orig = t
        m = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)([\.\)]\s+)", t)
        if m:
            t = t[m.end() :].lstrip()
            continue
        m = re.match(r"^\((\d{1,3})\)\s+", t)
        if m:
            t = t[m.end() :].lstrip()
            continue
        m = re.match(r"^([a-zA-Z])([\.\)]\s+)", t)
        if m and len(t) > len(m.group(0)) + 2:
            t = t[m.end() :].lstrip()
            continue
        m = re.match(r"^([ivxlcdm]{1,8})([\.\)]\s+)", t, re.I)
        if m and _ROMAN_NUM.match(m.group(1)) and len(t) > len(m.group(0)) + 2:
            t = t[m.end() :].lstrip()
            continue
        if t == orig:
            break
    return t.strip()


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def looks_like_formula_or_markup(s: str) -> bool:
    """Heuristic: LaTeX-ish or delimiter-heavy fragments are not useful concept labels."""
    if not s or not s.strip():
        return True
    t = s.strip()
    if "$" in t or "\\(" in t or "\\[" in t:
        return True
    if re.search(r"\\[a-zA-Z]+\b", t):
        return True
    if "`" in t or ("<" in t and ">" in t):
        return True
    non_space = re.sub(r"\s", "", t)
    if not non_space:
        return True
    special = sum(1 for c in non_space if c in "{}$\\^_&|~")
    if special / len(non_space) > 0.28:
        return True
    if re.fullmatch(r"[\d\s.\-–—_,;:+/\\()]+", t):
        return True
    return False


def is_only_numbering_label(s: str) -> bool:
    t = strip_leading_numbering(s)
    t = re.sub(r"\s+", "", t)
    if not t:
        return True
    return bool(re.fullmatch(r"[\d.]+", t))


def is_noise_concept(text: str, *, mode: str = "strict") -> bool:
    """
    Return True if this string should not be stored or shown as a concept.

    mode:
      - ``glossary`` — min length 2; for list/table glossary lines.
      - ``strict`` — min length 3; for headings and bold.
    """
    raw = _nfc(text.strip())
    if not raw:
        return True
    min_len = _MIN_LEN_GLOSSARY if mode == "glossary" else _MIN_LEN_STRICT
    if len(raw) > _MAX_LEN:
        return True
    stripped = strip_leading_numbering(raw)
    core = stripped.strip()
    if not core:
        return True
    if len(core) < min_len:
        return True
    if looks_like_formula_or_markup(core):
        return True
    if is_only_numbering_label(core):
        return True
    key = re.sub(r"\s+", " ", core.lower())
    key = key.strip(".,;:!?-*•'\"«»()[]`")
    if not key:
        return True
    if key in _STOP:
        return True
    if key in {"main", "details", "setup", "context", "today", "next", "previous"}:
        return True
    if re.match(r"^aufgabe\s+zu\b", key):
        return True
    if re.match(r"^typische\s+(fehler|fehltritte|fragestellungen|missverständnisse)\b", key):
        return True
    for phrase in _BOILERPLATE_SUBSTRINGS:
        if phrase in key:
            return True
    parts = key.split()
    if len(parts) == 1 and parts[0] in _STOP:
        return True
    return False


def should_show_concept_in_ui(name: str) -> bool:
    """UI list: hide obvious junk (same spirit as extraction, slightly permissive for 2-letter terms)."""
    n = _nfc(name.strip())
    if not n:
        return False
    if looks_like_formula_or_markup(n):
        return False
    if is_only_numbering_label(n):
        return False
    return not is_noise_concept(n, mode="glossary")


def filter_concept_rows_for_display(
    rows: list[dict],
    cap: int = 28,
) -> tuple[list[dict], int, bool, bool]:
    """
    Dedupe by normalized key, drop noise, cap length.

    Returns (filtered_rows, original_count, all_filtered_out, hit_display_cap).
    """
    from app.services.concept_normalize import normalize_concept_key

    original = len(rows)
    seen: set[str] = set()
    out: list[dict] = []
    hit_cap = False
    for r in rows:
        name = (r.get("name") or "").strip()
        if not should_show_concept_in_ui(name):
            continue
        nk = normalize_concept_key(name)
        if not nk or nk in seen:
            continue
        seen.add(nk)
        out.append(r)
        if len(out) >= cap:
            hit_cap = True
            break
    all_filtered = original > 0 and len(out) == 0
    return out, original, all_filtered, hit_cap
