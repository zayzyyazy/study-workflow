"""Clean upload / PDF-derived titles for human-readable lecture & material names."""

from __future__ import annotations

import re
from pathlib import Path

_MAX_BASE_LEN = 46
_DATE_TOKEN = re.compile(r"\b20\d{2}[-_]\d{1,2}[-_]\d{1,2}\b", re.I)
_UUID_LIKE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
# Leading tokens like "DM12345" or "INFO2024001" before a real title
_LEADING_COURSE_CODE = re.compile(r"^[A-Za-z]{2,6}\d{4,}\s+", re.I)
# Common slide/doc export boilerplate at the start of filenames
_EXPORT_APP_PREFIX = re.compile(
    r"^(?:(?:microsoft\s+)?(?:powerpoint|word|excel)|"
    r"google\s+slides?|keynote|libreoffice\s+impress|canva|"
    r"powerpoint\s+presentation)\s*[-–—:_]*\s*",
    re.I,
)
# "VL05 Topic", "UE03 Graphs" style filename prefixes (slot is shown separately in UI)
_LEADING_SLOT_ABBREV = re.compile(
    r"^(?:VL|VO|LEC|UE|UB|PS|AB|TK)(?:_|\s)?(\d{1,2})\s+",
    re.I,
)


def _strip_leading_slot_abbrev(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return t
    m = _LEADING_SLOT_ABBREV.match(t)
    if m and len(t[m.end() :].strip()) >= 4:
        return t[m.end() :].strip()
    m2 = re.match(r"^(VL|VO|LEC|UE|UB|PS)(\d{1,2})(\s+)(.+)$", t, re.I)
    if m2 and len(m2.group(4).strip()) >= 4:
        return m2.group(4).strip()
    return t


def _strip_leading_course_code_noise(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return t
    t2 = _LEADING_COURSE_CODE.sub("", t).strip()
    if len(t2) < 4:
        return t
    return t2


def _strip_export_app_prefix(s: str) -> str:
    t = (s or "").strip()
    for _ in range(4):
        n = _EXPORT_APP_PREFIX.sub("", t).strip()
        if n == t:
            break
        t = n
    return t


def _space_camel_case(s: str) -> str:
    """Turn 'InfoProcessing' / 'SetPractice' into spaced words for scanning."""
    t = (s or "").strip()
    if not t or " " in t or len(t) < 6:
        return t
    if not any(c.islower() for c in t) or not any(c.isupper() for c in t):
        return t
    spaced = re.sub(r"([a-zäöü0-9])([A-ZÄÖÜ])", r"\1 \2", t)
    if spaced != t:
        spaced = re.sub(r"([A-ZÄÖÜ]{2,})([A-ZÄÖÜ][a-zäöü])", r"\1 \2", spaced)
    return re.sub(r"\s+", " ", spaced).strip()


def scrub_filename_stem(stem: str) -> str:
    """Strip extension artifacts and noisy filename tokens; keep semantic words."""
    s = (stem or "").strip()
    if not s:
        return ""
    s = Path(s).stem if "." in s else s
    s = s.replace("\u00a0", " ")
    s = _UUID_LIKE.sub(" ", s)
    dates = _DATE_TOKEN.findall(s)
    if len(dates) >= 2 and len({d.lower().replace("_", "-") for d in dates}) == 1:
        s = _DATE_TOKEN.sub(" ", s, count=1)
    s = _DATE_TOKEN.sub(" ", s)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _strip_export_app_prefix(s)
    s = _strip_leading_course_code_noise(s)
    s = _strip_leading_slot_abbrev(s)
    for _ in range(6):
        old = s
        s = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", s).strip()
        s = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", s).strip()
        s = re.sub(r"_\d+$", "", s).strip()
        s = re.sub(r"\s-\s*\d+\s*$", "", s).strip()
        s = re.sub(r"(?i)\s*(?:copy|kopie|duplikat|final|druck|export|scan)\d*\s*$", "", s).strip()
        s = re.sub(r"(?i)\s*v\d+\s*$", "", s).strip()
        if s == old:
            break
    s = re.sub(r"^[\d\s.\-–—]+", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    s = _space_camel_case(s)
    return s


def _digit_ratio(t: str) -> float:
    letters = sum(1 for c in t if c.isalpha())
    digits = sum(1 for c in t if c.isdigit())
    if letters + digits == 0:
        return 1.0
    return digits / (letters + digits + 0.01)


def title_quality_score(text: str) -> float:
    """Higher = more likely a good human title (not raw filename)."""
    t = (text or "").strip()
    if len(t) < 6:
        return 0.0
    score = 12.0
    score += min(len(t), 72) * 0.25
    if _digit_ratio(t) > 0.45:
        score -= 18.0
    if re.search(r"\b\d{6,}\b", t):
        score -= 10.0
    if len(t.split()) > 16:
        score -= 8.0
    words = [w for w in t.split() if len(w) > 1]
    if len(words) >= 3:
        score += 8.0
    if re.search(r"(?:,|–|—| und | and | or )\s+\w", t, re.I):
        score += 4.0
    return score


def prefer_metadata_or_stem(metadata: str, stem_scrubbed: str) -> str:
    meta = (metadata or "").strip()
    stem = (stem_scrubbed or "").strip()
    if not meta:
        return stem
    if not stem:
        return meta
    mq = title_quality_score(meta)
    sq = title_quality_score(stem)
    if mq >= sq + 3.0 and len(meta) <= 120:
        return meta
    if mq >= sq + 1.0 and len(meta) >= 10 and _digit_ratio(meta) < 0.35:
        return meta
    return stem


def strip_redundant_material_prefix(base: str, material_kind: str) -> str:
    """
    Remove leading 'Lecture 3 - …' / 'Sheet 2 …' when the UI already prefixes
    Lecture/Sheet/Material + slot index.
    """
    s = (base or "").strip()
    if not s:
        return s
    patterns = [
        r"^(?:lecture|lec|vorlesung|folien|slides?)\s*\d{1,2}\s*[:\-–]\s*",
        r"^(?:lecture|lec|vorlesung|folien|slides?)\s+\d{1,2}\s+",
    ]
    if material_kind == "exercise":
        patterns = [
            r"^(?:sheet|blatt|übung|uebung|aufgabe|exercise|homework|problem\s*set)\s*\d{1,2}\s*[:\-–]\s*",
            r"^(?:sheet|blatt|übung|uebung|aufgabe|exercise)\s+\d{1,2}\s+",
        ] + patterns
    if material_kind == "material":
        patterns = [r"^(?:material|handout|reader|skript)\s*[:\-–]\s*"] + patterns
    for _ in range(4):
        old = s
        for pat in patterns:
            s = re.sub(pat, "", s, flags=re.I).strip()
        if s == old:
            break
    s = _space_camel_case(s)
    return s


def polish_readable_base(s: str, *, max_len: int = _MAX_BASE_LEN) -> str:
    """Final spacing, length cap, no trailing junk."""
    t = (s or "").strip()
    if not t:
        return t
    t = re.sub(r"\s+", " ", t)
    t = _space_camel_case(t)
    t = re.sub(r"\s*[-–—,;:]+\s*$", "", t).strip()
    if len(t) > max_len:
        cut = t[: max_len + 1]
        if " " in cut:
            t = cut.rsplit(" ", 1)[0].strip()
        else:
            t = cut[:max_len].rstrip(" -–—.,;:")
    return t
