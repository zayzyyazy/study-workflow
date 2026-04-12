"""
Heuristic analysis of extracted lecture text for language and content style.

No extra LLM calls — deterministic rules only (German vs English, math/code signals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

LanguageCode = Literal["de", "en"]
ContentProfile = Literal["general", "math", "code", "mixed"]

# Common function words (small sets — enough to bias de vs en on lecture prose)
_GERMAN_HINTS = frozenset(
    {
        "der",
        "die",
        "das",
        "und",
        "nicht",
        "ist",
        "ein",
        "eine",
        "für",
        "von",
        "mit",
        "auf",
        "als",
        "auch",
        "nach",
        "über",
        "werden",
        "haben",
        "sein",
        "sich",
        "noch",
        "nur",
        "oder",
        "bei",
        "wie",
        "wird",
        "dem",
        "den",
        "des",
        "im",
        "zum",
        "zur",
        "dass",
        "kann",
        "können",
        "müssen",
        "wenn",
        "dann",
    }
)
_ENGLISH_HINTS = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "for",
        "that",
        "with",
        "on",
        "as",
        "are",
        "was",
        "were",
        "be",
        "this",
        "which",
        "from",
        "at",
        "or",
        "an",
        "by",
        "not",
        "have",
        "has",
        "will",
        "can",
        "if",
        "then",
        "when",
        "than",
        "such",
        "their",
    }
)

_MATH_PATTERNS = [
    re.compile(r"\\[a-zA-Z]+"),  # LaTeX commands
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"\$[^$]+\$"),
    re.compile(r"\\begin\{"),
    re.compile(r"[∫∑∏√∞≤≥≠≈∈∉⊂⊆∀∃]"),
    re.compile(r"\^[^{]"),
    re.compile(r"_\{"),
    re.compile(r"\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma"),
]
_CODE_PATTERNS = [
    re.compile(r"^```", re.MULTILINE),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w+\s+import\b"),
    re.compile(r"\bfunction\s+\w+\s*\("),
    re.compile(r"\b(public|private|static)\s+(class|void|int)\b"),
    re.compile(r";\s*$", re.MULTILINE),
    re.compile(r"\{\s*\n"),
]


@dataclass
class LectureAnalysis:
    detected_language: LanguageCode
    content_profile: ContentProfile
    has_formulas: bool
    has_code: bool
    notes: str

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "content_profile": self.content_profile,
            "has_formulas": self.has_formulas,
            "has_code": self.has_code,
            "notes": self.notes,
            "analysis_updated_at": datetime.now(timezone.utc).isoformat(),
        }


def _words_lower(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower())


def _detect_language(text: str) -> LanguageCode:
    if len(text.strip()) < 80:
        # Very short: umlauts strongly suggest German
        if re.search(r"[äöüÄÖÜß]", text):
            return "de"
        return "en"

    words = _words_lower(text[:80000])
    if not words:
        return "en"

    de_hits = sum(1 for w in words if w in _GERMAN_HINTS)
    en_hits = sum(1 for w in words if w in _ENGLISH_HINTS)
    umlaut_bonus = sum(text.count(c) for c in "äöüßÄÖÜ") * 2

    de_score = de_hits + umlaut_bonus
    en_score = en_hits

    # Tie-break toward dominant token count
    if de_score >= max(8, en_score * 1.12):
        return "de"
    if en_score >= max(8, de_score * 1.12):
        return "en"
    # Ambiguous: more German function words or umlauts?
    if de_score > en_score or umlaut_bonus >= 3:
        return "de"
    return "en"


def _math_score(text: str) -> float:
    s = 0.0
    for pat in _MATH_PATTERNS:
        s += len(pat.findall(text)) * 1.0
    # Lines with multiple = or Unicode math
    for line in text.splitlines():
        if line.count("=") >= 2 and len(line) < 200:
            s += 0.5
    return s


def _code_score(text: str) -> float:
    s = 0.0
    for pat in _CODE_PATTERNS:
        s += len(pat.findall(text)) * 1.0
    if text.count("```") >= 2:
        s += 4.0
    return s


def _pick_profile(math_s: float, code_s: float) -> tuple[ContentProfile, bool, bool]:
    has_math = math_s >= 2.5
    has_code = code_s >= 2.5
    strong_math = math_s >= 6.0
    strong_code = code_s >= 6.0

    if strong_math and strong_code:
        return "mixed", True, True
    if strong_math:
        return "math", True, has_code
    if strong_code:
        return "code", has_math, True
    if has_math and has_code:
        return "mixed", True, True
    if has_math:
        return "math", True, False
    if has_code:
        return "code", False, True
    return "general", bool(math_s >= 0.5), bool(code_s >= 0.5)


def analyze_extracted_text(text: str) -> LectureAnalysis:
    """
    Analyze truncated or full extracted lecture text.
    """
    sample = text if len(text) <= 120_000 else text[:120_000]
    lang = _detect_language(sample)
    ms = _math_score(sample)
    cs = _code_score(sample)
    profile, hf, hc = _pick_profile(ms, cs)
    notes = f"heuristic math_score={ms:.1f} code_score={cs:.1f} lang_tokens"
    return LectureAnalysis(
        detected_language=lang,
        content_profile=profile,
        has_formulas=hf,
        has_code=hc,
        notes=notes,
    )


def analysis_from_meta(meta: dict[str, Any]) -> LectureAnalysis | None:
    """Rebuild analysis from meta.json lecture_analysis block if present."""
    block = meta.get("lecture_analysis")
    if not isinstance(block, dict):
        return None
    try:
        lang = block.get("detected_language", "en")
        if lang not in ("de", "en"):
            lang = "en"
        prof = block.get("content_profile", "general")
        if prof not in ("general", "math", "code", "mixed"):
            prof = "general"
        return LectureAnalysis(
            detected_language=lang,
            content_profile=prof,
            has_formulas=bool(block.get("has_formulas")),
            has_code=bool(block.get("has_code")),
            notes=str(block.get("notes") or ""),
        )
    except (TypeError, ValueError):
        return None
