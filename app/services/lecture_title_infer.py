"""Infer a human lecture title from extracted PDF/slide text (deterministic, no AI)."""

from __future__ import annotations

import re
# Head window: title slides usually appear early
_HEAD_CHARS = 14000
# Reject obvious non-titles; card-friendly candidates stay shorter
_MAX_LINE_LEN = 88
_MAX_PLAIN_LINE_FOR_EXERCISE = 64
_MIN_TITLE_LEN = 8
_MIN_TITLE_LEN_EXERCISE = 5
# Strong preference for title-like lines in the first ~2 slides worth of text
_EARLY_LINE_CAP = 48
# Hard cap for the base segment shown after "Lecture 02 - …" (cards / lists)
_CARD_BASE_MAX_CHARS = 46
_CARD_BASE_MAX_WORDS = 7

# German / English exercise boilerplate — not scan-friendly as a title
_EXERCISE_INSTRUCTION_START = re.compile(
    r"(?i)^[\s\(（]*(?:[a-zäöü0-9]+[\)）]\s*)*"
    r"(?:"
    r"gegeben\s+(?:sind|sei|seien)|"
    r"bearbeiten\s+sie|"
    r"zeigen\s+sie|"
    r"bestimmen\s+sie|"
    r"berechnen\s+sie|"
    r"lösen\s+sie|"
    r"beweisen\s+sie|"
    r"wählen\s+sie|"
    r"füllen\s+sie|"
    r"ordnen\s+sie|"
    r"untersuchen\s+sie|"
    r"stellen\s+sie|"
    r"es\s+seien|es\s+sei|"
    r"betrachten\s+sie|"
    r"seien?\s+(?:die\s+)?mengen?\b|"
    r"gegeben\s+ist|"
    r"given\s+(?:the|that)|"
    r"show\s+that|prove\s+that|determine\s+|"
    r"find\s+(?:the|all|a)\b|"
    r"suppose\s+that|"
    r"let\s+[A-Z]\s+be\b|"
    r"solve\s+(?:the|all)\b|"
    r"compute\s+|calculate\s+"
    r")",
)

_MATHY_OR_TASK_FRAGMENT = re.compile(
    r"(=|\{|\}|\[[\d,\s]{6,}\]|"
    r"\bA\s*=\s*\{|\bB\s*=\s*\{|\bM\s*=\s*\{|\bU\s*=\s*\{|"
    r"\(\s*c\s*\)\s*gegeben|"
    r"©|\(c\))",
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


def _strip_leading_enum_marker(s: str) -> str:
    """Remove '(c)', 'a)', '1.' style markers at the start."""
    t = (s or "").strip()
    for _ in range(4):
        n = re.sub(
            r"^[\s#•\-–—]*(?:[\(（]\s*[a-zA-ZäöüÄÖÜ0-9]+\s*[\)）]|[a-zA-Zäöü]\s*[\).:])\s*",
            "",
            t,
        ).strip()
        n = re.sub(r"^\d{1,2}\s*[\).:]\s*", "", n).strip()
        if n == t:
            break
        t = n
    return t


def looks_like_exercise_instruction(text: str) -> bool:
    """True if this looks like task text, not a sheet topic heading."""
    s = _strip_leading_enum_marker(text or "")
    if not s:
        return False
    if _EXERCISE_INSTRUCTION_START.search(s):
        return True
    if _MATHY_OR_TASK_FRAGMENT.search(s):
        return True
    if s.count("=") >= 2:
        return True
    if "{" in s and "}" in s and re.search(r"\d\s*,\s*\d", s):
        return True
    low = s.lower()
    if "aufgabe" in low and re.search(r"aufgabe\s*\d", low):
        if len(s) > 40:
            return True
    return False


def looks_like_sentence_dump(text: str, *, material_kind: str) -> bool:
    """Long imperative / run-on lines are poor card titles."""
    s = (text or "").strip()
    if len(s) > 58:
        return True
    if material_kind == "exercise" and len(s) > 48:
        return True
    if s.count(",") >= 3 and len(s) > 50:
        return True
    if s.count(";") >= 1:
        return True
    if re.search(r"\b(sind|seien|sind\s+die|that|which|where)\b", s, re.I) and len(s) > 36:
        return True
    return False


def squeeze_card_base(text: str, *, material_kind: str) -> str:
    """
    Turn a long heading or fragment into a short topic label for UI cards.
    """
    s = _strip_leading_enum_marker(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    # Prefer clause before comma for long German/English titles
    if len(s) > 44 and "," in s:
        first = s.split(",", 1)[0].strip()
        if len(first) >= 6 and len(first) <= _CARD_BASE_MAX_CHARS + 8:
            s = first
        elif len(first) < 6:
            pass
        else:
            s = first
    words = s.split()
    if len(words) > _CARD_BASE_MAX_WORDS:
        s = " ".join(words[:_CARD_BASE_MAX_WORDS])
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > _CARD_BASE_MAX_CHARS:
        cut = s[: _CARD_BASE_MAX_CHARS + 1]
        if " " in cut:
            s = cut.rsplit(" ", 1)[0].strip()
        else:
            s = cut[:_CARD_BASE_MAX_CHARS].rstrip(" -–—.,;:")
    s = re.sub(r"\s*[-–—,;:]+\s*$", "", s).strip()
    return s


def is_unacceptable_card_title(text: str, material_kind: str) -> bool:
    """If true, prefer filename/metadata fallback over this candidate."""
    if not (text or "").strip():
        return True
    if looks_like_exercise_instruction(text):
        return True
    if looks_like_sentence_dump(text, material_kind=material_kind):
        return True
    if len(text) > _CARD_BASE_MAX_CHARS + 8:
        return True
    return False


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
    return s[:_MAX_LINE_LEN]


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
_SHEET_HEADING = re.compile(
    r"^(?:sheet|übungsblatt|uebungsblatt|aufgabenblatt|homework|problem\s*set|exercise)"
    r"(?:\s+\d{1,2})?\s*[:\-–—]\s*(.+)$",
    re.I,
)


def infer_base_title_from_extracted_text(
    head_text: str,
    *,
    fallback: str,
    material_kind: str = "lecture",
) -> str:
    """
    Best-effort title from the start of extracted text.
    Returns fallback if nothing reliable is found.
    """
    kind = (material_kind or "lecture").strip().lower()
    if kind not in ("lecture", "exercise", "material"):
        kind = "lecture"

    text = (head_text or "")[:_HEAD_CHARS]
    if not text.strip():
        return _post_process_candidate(fallback, kind, is_fallback=True)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    scored: list[tuple[float, str]] = []

    def add_candidate(raw: str, *, line_idx: int, base: float, source: str) -> None:
        c = _clean_title_candidate(raw)
        if not c or _looks_like_noise(c):
            return
        if looks_like_exercise_instruction(c):
            return
        if kind == "exercise":
            if source == "plain" and (
                len(c) > _MAX_PLAIN_LINE_FOR_EXERCISE
                or looks_like_sentence_dump(c, material_kind=kind)
            ):
                return
        elif kind == "material":
            if source == "plain" and len(c) > 72:
                return
        if looks_like_sentence_dump(c, material_kind=kind) and source == "plain":
            return

        score = base + min(len(c), 64) * 0.12
        if line_idx < 12:
            score += 18.0
        elif line_idx < _EARLY_LINE_CAP:
            score += 10.0
        if re.search(r"(?:und|and|oder)\s+", c, re.I):
            score += 4.0
        words = c.split()
        if len(words) >= 3:
            score += 5.0
        if len(c) > 52:
            score -= 6.0
        if kind == "exercise" and source == "plain":
            score -= 10.0
        scored.append((score, c))

    for line_idx, ln in enumerate(lines[:90]):
        m = re.match(r"^#{1,3}\s+(.+)$", ln)
        if m:
            add_candidate(m.group(1), line_idx=line_idx, base=14.0, source="heading")
        m2 = _LECTURE_NUM.match(ln)
        if m2:
            add_candidate(m2.group(2), line_idx=line_idx, base=22.0, source="lecture")
        m3 = _LECTURE_NUM_TIGHT.match(ln)
        if m3:
            add_candidate(m3.group(2), line_idx=line_idx, base=20.0, source="lecture")
        tc = _TITLE_COLON.match(ln)
        if tc:
            add_candidate(tc.group(1), line_idx=line_idx, base=16.0, source="meta")
        sh = _SHEET_HEADING.match(ln)
        if sh and kind == "exercise":
            add_candidate(sh.group(1).strip(), line_idx=line_idx, base=18.0, source="sheet")

        if len(ln) < 120 and not ln.startswith("#"):
            plain = _clean_title_candidate(re.sub(r"^\d+\.\s*", "", ln))
            min_len = _MIN_TITLE_LEN_EXERCISE if kind == "exercise" else _MIN_TITLE_LEN
            if plain and len(plain) >= min_len:
                if re.search(r"[A-ZÄÖÜa-zäöü]", plain):
                    add_candidate(plain, line_idx=line_idx, base=4.0, source="plain")

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
        min_ok = _MIN_TITLE_LEN_EXERCISE if kind == "exercise" else _MIN_TITLE_LEN
        if len(best) >= min_ok:
            out = _post_process_candidate(best, kind, is_fallback=False)
            if not is_unacceptable_card_title(out, kind):
                return out

    return _post_process_candidate(fallback, kind, is_fallback=True)


def _safe_short_fallback(material_kind: str) -> str:
    if material_kind == "exercise":
        return "Exercises"
    if material_kind == "material":
        return "Reading"
    return "Topics"


def _post_process_candidate(s: str, material_kind: str, *, is_fallback: bool) -> str:
    t = squeeze_card_base(s, material_kind=material_kind)
    t = _normalize_title_case(t)
    if not t.strip():
        return _safe_short_fallback(material_kind)
    if is_fallback and is_unacceptable_card_title(t, material_kind):
        return _safe_short_fallback(material_kind)
    return t
