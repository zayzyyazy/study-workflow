"""Heuristic course + material-type inference from PDFs (no AI)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.services import extraction_service

MaterialKind = Literal["lecture", "exercise", "material"]

_STOP = frozenset(
    """
    der die das und oder the for from with ohne eine einem einen einer eines ein
    von aus bei zum zur zum dem den des im in an auf ist sind war wird wer wie
    was wann kann nicht nur mehr auch noch als wenn dann this that these those
    page seite folie slide slides lecture week unit chapter kapitel
    """.split()
)


def _norm_token(raw: str) -> str:
    s = unicodedata.normalize("NFKD", raw)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    if len(s) < 3:
        return ""
    if s in _STOP:
        return ""
    return s


def tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"[0-9A-Za-zÄÖÜäöüß]{3,}", text or ""):
        t = _norm_token(m.group(0))
        if t:
            out.add(t)
    return out


def read_pdf_metadata_title(path: Path) -> str:
    """Public wrapper: document Title metadata from PDF, or empty string."""
    return _pdf_metadata_title(path)


def _pdf_metadata_title(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        meta = reader.metadata
        if not meta:
            return ""
        for key in ("/Title", "Title"):
            v = meta.get(key) if hasattr(meta, "get") else None
            if v is None and hasattr(meta, "title"):
                v = getattr(meta, "title", None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _head_text(path: Path, max_chars: int = 12000) -> str:
    ext = extraction_service.extract_text_from_file(path)
    if not ext.ok:
        return ""
    return (ext.text or "")[:max_chars]


_KIND_PATTERNS: dict[MaterialKind, tuple[str, ...]] = {
    "exercise": (
        r"\baufgaben",
        r"\bübung",
        r"\buebung",
        r"\bblatt\b",
        r"\bworksheet",
        r"\bhomework",
        r"\bproblem\s*set",
        r"\btutorial\b",
        r"\btutorium",
        r"\bpractice\b",
        r"\bexercises?\b",
        r"\bweekly\s*quiz",
        r"\bquiz\b",
        r"\bserie\b",
        r"\bsheet\b",
    ),
    "lecture": (
        r"\blecture\b",
        r"\bvorlesung\b",
        r"\bvl\b",
        r"\bslides?\b",
        r"\bfolien\b",
        r"\bpresentation\b",
        r"\bweek\s*\d",
        r"\bsession\b",
        r"\bseminar\b",
        r"\bkapitel\b",
        r"\bchapter\b",
    ),
    "material": (
        r"\breader\b",
        r"\bskript\b",
        r"\bhandout\b",
        r"\bmaterial\b",
        r"\bnotes\b",
        r"\bnotizen\b",
        r"\bcourse\s*pack\b",
        r"\bappendix\b",
        r"\banhang\b",
        r"\bsyllabus\b",
        r"\boutline\b",
    ),
}


def classify_material_kind(*, filename: str, pdf_title: str, head_text: str) -> tuple[MaterialKind, dict[str, float], str]:
    blob = f"{filename}\n{pdf_title}\n{head_text[:8000]}".lower()
    scores: dict[str, float] = {k: 0.0 for k in ("lecture", "exercise", "material")}
    for kind, pats in _KIND_PATTERNS.items():
        for pat in pats:
            for m in re.finditer(pat, blob, flags=re.I):
                scores[kind] += 2.5 if pat.startswith(r"\baufgaben") else 1.8
    # Filename emphasis
    stem = Path(filename or "").stem.lower()
    for kind, pats in _KIND_PATTERNS.items():
        for pat in pats:
            if re.search(pat, stem, flags=re.I):
                scores[kind] += 3.0

    # Weak default: treat as lecture when slide-like length / structure
    if max(scores.values()) < 1.0:
        if re.search(r"\bfolie\s*\d|\bslide\s*\d", blob[:4000], re.I):
            scores["lecture"] += 2.0
        if re.search(r"\baufgabe\s*\d|\bexercise\s*\d", blob[:6000], re.I):
            scores["exercise"] += 2.5

    best_key = max(scores, key=lambda k: scores[k])
    second = sorted(scores.values(), reverse=True)[1]
    margin = scores[best_key] - second
    if scores[best_key] < 0.5:
        chosen: MaterialKind = "lecture"
        note = "No strong type signals; defaulting to lecture."
    else:
        chosen = best_key  # type: ignore[assignment]
        note = f"Type signals favor {chosen} (score {scores[best_key]:.1f}, margin {margin:.1f})."
    return chosen, scores, note


def _material_confidence(scores: dict[str, float], chosen: str) -> float:
    total = sum(max(0.0, v) for v in scores.values()) + 0.01
    top = max(scores.values())
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    base = scores.get(chosen, 0.0) / total
    if top <= 0.01:
        return 0.35
    gap = (top - second) / (top + 0.01)
    return min(0.97, max(0.35, 0.45 * base + 0.45 * gap))


def _course_context_tokens(course: dict[str, Any], lecture_titles: list[str]) -> set[str]:
    parts = [course.get("name", ""), str(course.get("slug", "")).replace("-", " ")]
    parts.extend(lecture_titles[:12])
    return tokenize("\n".join(parts))


def rank_courses(
    courses: list[dict[str, Any]],
    lecture_titles_by_course: dict[int, list[str]],
    doc_tokens: set[str],
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for c in courses:
        cid = int(c["id"])
        ctx = _course_context_tokens(c, lecture_titles_by_course.get(cid, []))
        inter = len(doc_tokens & ctx)
        # substring boosts for short codes (e.g. "dm", "psy")
        bonus = 0.0
        cname = str(c.get("name", "")).lower()
        cslug = str(c.get("slug", "")).lower().replace("-", " ")
        for tok in doc_tokens:
            if len(tok) <= 5 and tok and (tok in cname or tok in cslug):
                bonus += 1.2
        score = float(inter) * 2.1 + bonus
        ranked.append(
            (
                score,
                {
                    "course_id": cid,
                    "course_name": c["name"],
                    "score": round(score, 2),
                    "overlap_tokens": sorted(doc_tokens & ctx)[:16],
                },
            )
        )
    ranked.sort(key=lambda x: -x[0])
    return [r[1] for r in ranked]


def course_confidence_from_ranked(ranked: list[dict[str, Any]]) -> tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    top = float(ranked[0]["score"])
    second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    conf = top / (top + second + 1.0) if top > 0 else 0.0
    return min(0.98, conf), top - second


def should_auto_place_course(conf: float, margin: float, top_score: float) -> bool:
    if top_score <= 0:
        return False
    return conf >= 0.72 and margin >= 2.0


def should_auto_place_kind(material_conf: float, scores: dict[str, float]) -> bool:
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return material_conf >= 0.55
    return material_conf >= 0.58 and (vals[0] - vals[1]) >= 1.5


@dataclass
class IntakeAnalysis:
    pdf_title: str
    head_text: str
    material_kind: MaterialKind
    material_scores: dict[str, float]
    material_note: str
    material_confidence: float
    auto_place_kind: bool
    ranked_courses: list[dict[str, Any]]
    course_confidence: float
    course_margin: float
    auto_place_course: bool
    doc_tokens_sample: list[str]


def analyze_pdf_for_intake(
    path: Path,
    *,
    courses: list[dict[str, Any]],
    lecture_titles_by_course: dict[int, list[str]],
    original_filename: str | None = None,
) -> IntakeAnalysis:
    filename = (original_filename or path.name).strip() or path.name
    pdf_title = _pdf_metadata_title(path)
    head = _head_text(path)
    doc_tokens = tokenize(f"{filename}\n{pdf_title}\n{head[:6000]}")
    kind, mscores, mnote = classify_material_kind(filename=filename, pdf_title=pdf_title, head_text=head)
    mconf = _material_confidence(mscores, kind)
    auto_k = should_auto_place_kind(mconf, mscores)
    ranked = rank_courses(courses, lecture_titles_by_course, doc_tokens)
    cconf, margin = course_confidence_from_ranked(ranked)
    top_score = float(ranked[0]["score"]) if ranked else 0.0
    auto_c = should_auto_place_course(cconf, margin, top_score)
    return IntakeAnalysis(
        pdf_title=pdf_title,
        head_text=head,
        material_kind=kind,
        material_scores=mscores,
        material_note=mnote,
        material_confidence=round(mconf, 3),
        auto_place_kind=auto_k,
        ranked_courses=ranked,
        course_confidence=round(cconf, 3),
        course_margin=round(margin, 3),
        auto_place_course=auto_c,
        doc_tokens_sample=sorted(doc_tokens)[:24],
    )
