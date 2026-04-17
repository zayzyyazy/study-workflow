"""
Heuristic analysis of extracted lecture text for language and content style.

No extra LLM calls — deterministic rules only (German vs English, math/code signals,
lecture kind, depth band, organizational vs technical, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

LanguageCode = Literal["de", "en"]
ContentProfile = Literal["general", "math", "code", "mixed"]
LectureKind = Literal[
    "organizational",
    "conceptual",
    "mathematical",
    "proof_heavy",
    "coding",
    "mixed",
    "general",
]
DepthBand = Literal["light", "medium", "dense"]
SourceGroundingStrength = Literal["low", "medium", "high"]
TopicGranularity = Literal["coarse", "medium", "fine"]
FormalDensity = Literal["low", "medium", "high"]
ConceptualDensity = Literal["low", "medium", "high"]

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

# Logistics / admin — exams, platforms, deadlines (DE + EN)
_ORG_PATTERNS = [
    re.compile(
        r"\b(prüfung|klausur|nachklausur|exam|midterm|final|quiz|test|abgabe|deadline|"
        r"assignment|homework|hausaufgabe|moodle|ecampus|ilias|stud\.ip|syllabus|"
        r"organisatorisch|organisatorisches|teilnahme|anwesenheit|anwesenheitspflicht|"
        r"sprechstunde|office hours|piazza|canvas|blackboard|turnitin|"
        r"semester|wochenplan|terminplan|frist|due date|einschreibung|anmeldung|"
        r"credit points|ects|leistungspunkte|bewertung|noten|grading|"
        r"course policy|academic integrity|plagiarism)\b",
        re.I,
    ),
    re.compile(
        r"\b(please submit|submit by|due:|due on|readings for|next week we|"
        r"important dates|course schedule|tentative schedule)\b",
        re.I,
    ),
    re.compile(r"\d{1,2}[./]\d{1,2}([./]\d{2,4})?"),  # dates like 12.04 or 12/04/26
]

# Proof-style wording (DE + EN)
_PROOF_PATTERNS = [
    re.compile(
        r"\b(beweis|beweise|bewiesen|beweisen|zu zeigen|zz\.|w\.z\.b\.w\.|"
        r"theorem|theorems|lemma|lemmas|korollar|corollary|proposition|"
        r"proof|q\.e\.d\.|qed|folgt aus|folglich|annahme|annehmen|"
        r"induktion|induktionsschritt|widerspruch|contradiction|"
        r"beliebig|arbitrary)\b",
        re.I,
    ),
]

# Definition-heavy / conceptual (not math-specific)
_DEF_PATTERNS = [
    re.compile(
        r"\b(definition|definiert|definiere|bezeichnet|bedeutet|notion|konzept|"
        r"framework|intuition|intuitiv|im gegensatz|vergleich|unterscheidung|"
        r"implies|therefore|hence|thus|concept|concepts|distinction)\b",
        re.I,
    ),
]

# Examples / exercises frequency
_EXAMPLE_PATTERNS = [
    re.compile(
        r"\b(beispiel|beispiele|zum beispiel|example|examples|for example|"
        r"übung|übungen|exercise|exercises|aufgabe|aufgaben|"
        r"worked example|sample solution)\b",
        re.I,
    ),
]

# Markdown-ish headings (lines starting with #)
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)

_VALID_KINDS = frozenset(
    {
        "organizational",
        "conceptual",
        "mathematical",
        "proof_heavy",
        "coding",
        "mixed",
        "general",
    }
)
_VALID_DEPTH = frozenset({"light", "medium", "dense"})
_VALID_GROUND = frozenset({"low", "medium", "high"})
_VALID_GRANULARITY = frozenset({"coarse", "medium", "fine"})
PracticalDensity = Literal["low", "medium", "high"]

# Problem-solving / exercise-style wording (combined text may include "## Source:" exercise sheets)
_EXERCISE_SHEET_MARKERS = re.compile(
    r"^\*\*Role:\*\*\s*exercise",
    re.MULTILINE | re.IGNORECASE,
)
_TASK_LANGUAGE = [
    re.compile(
        r"\b(aufgabe|aufgaben|übung|übungen|übungsblatt|übungsblätter|tutorium|"
        r"exercise|exercises|problem\s*set|homework|assignment|klausuraufgabe|"
        r"zeige\s+dass|beweise|show\s+that|prove\s+that|calculate|berechne|"
        r"gegeben\s+sei|given)\b",
        re.I,
    ),
    re.compile(r"^\s*\(?[a-z0-9]+\)\s+", re.MULTILINE),  # (a) (b) style
]


@dataclass
class LectureAnalysis:
    detected_language: LanguageCode
    content_profile: ContentProfile
    has_formulas: bool
    has_code: bool
    notes: str
    lecture_kind: LectureKind
    depth_band: DepthBand
    is_organizational: bool
    is_proof_heavy: bool
    has_exercise_material: bool
    practical_density: PracticalDensity
    problem_solving_emphasis: bool
    source_grounding_strength: SourceGroundingStrength
    topic_granularity: TopicGranularity
    formal_density: FormalDensity
    conceptual_density: ConceptualDensity

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "content_profile": self.content_profile,
            "has_formulas": self.has_formulas,
            "has_code": self.has_code,
            "notes": self.notes,
            "lecture_kind": self.lecture_kind,
            "depth_band": self.depth_band,
            "is_organizational": self.is_organizational,
            "is_proof_heavy": self.is_proof_heavy,
            "has_exercise_material": self.has_exercise_material,
            "practical_density": self.practical_density,
            "problem_solving_emphasis": self.problem_solving_emphasis,
            "source_grounding_strength": self.source_grounding_strength,
            "topic_granularity": self.topic_granularity,
            "formal_density": self.formal_density,
            "conceptual_density": self.conceptual_density,
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


def _pattern_hits(patterns: list[re.Pattern[str]], text: str) -> float:
    return float(sum(len(p.findall(text)) for p in patterns))


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


def _depth_band(sample: str, math_s: float, heading_lines: int) -> DepthBand:
    n = max(len(sample), 1)
    math_per_1k = math_s / (n / 1000.0)
    if n < 4000 and math_per_1k < 2.5 and heading_lines < 7:
        return "light"
    if n > 38000 or math_per_1k > 14.0 or heading_lines > 34:
        return "dense"
    if n > 22000 and (math_per_1k > 8.0 or heading_lines > 22):
        return "dense"
    return "medium"


def _classify_lecture_kind(
    *,
    profile: ContentProfile,
    math_s: float,
    code_s: float,
    org_hits: float,
    proof_hits: float,
    def_hits: float,
    ex_hits: float,
    heading_lines: int,
    n_chars: int,
) -> tuple[LectureKind, bool, bool]:
    """
    Return (lecture_kind, is_organizational, is_proof_heavy).
    Order: organizational → proof-heavy → mixed (math+code) → coding → mathematical → conceptual → general.
    """
    strong_math = math_s >= 6.0
    strong_code = code_s >= 6.0
    med_math = math_s >= 3.0
    med_code = code_s >= 3.0

    # Organizational: many logistics signals, low formal density (avoid false positives on math courses)
    org_strong = org_hits >= 12.0 and math_s < 5.5 and code_s < 5.5
    org_dom = org_hits >= 8.0 and org_hits >= proof_hits + 5.0 and math_s < 6.0 and code_s < 6.0
    org_short = n_chars < 6000 and org_hits >= 6.0 and math_s < 3.0 and code_s < 3.0
    if org_strong or org_dom or org_short:
        return "organizational", True, False

    # Proof-heavy: explicit proof language + some mathematical content
    proof_strong = proof_hits >= 8.0 and med_math
    proof_ratio = proof_hits >= 5.0 and proof_hits >= org_hits * 0.9 and math_s >= 2.5
    if proof_strong or proof_ratio:
        return "proof_heavy", False, True

    # Mixed technical: both math and code signals (balance)
    if profile == "mixed" or (med_math and med_code and strong_math and strong_code):
        return "mixed", False, False
    if med_math and med_code and min(math_s, code_s) >= 4.0:
        return "mixed", False, False

    # Coding-first
    if profile == "code" or (strong_code and math_s < max(5.0, code_s * 0.85)):
        return "coding", False, False

    # Mathematical
    if profile == "math" or strong_math:
        return "mathematical", False, False

    # Conceptual / theory: definitions and distinctions, not formula-dense
    if math_s < 4.0 and code_s < 4.0 and def_hits >= 10.0 and heading_lines >= 4:
        return "conceptual", False, False
    if math_s < 3.5 and code_s < 3.5 and def_hits >= 6.0 and ex_hits < def_hits * 0.5:
        return "conceptual", False, False

    return "general", False, False


def _structural_signals(
    sample: str,
    *,
    math_s: float,
    proof_hits: float,
    def_hits: float,
    heading_lines: int,
) -> tuple[SourceGroundingStrength, TopicGranularity, FormalDensity, ConceptualDensity]:
    """
    Deterministic cues for grounding, topic structure, and formality — used in prompts only.
    """
    n = max(len(sample), 1)
    per10k = n / 10_000.0
    math_p10 = math_s / per10k
    proof_p10 = proof_hits / per10k
    def_p10 = def_hits / per10k
    h = heading_lines
    hp10 = h / per10k

    # How much structure/text we have to stay faithful to (thin source → narrow output)
    if n < 3500 or (h < 3 and n < 9000):
        sgs: SourceGroundingStrength = "low"
    elif n > 16_000 and h >= 9:
        sgs = "high"
    else:
        sgs = "medium"

    # Fine = many titled chunks / definition-rich; coarse = long text, few headings (broad slides)
    if h >= 10 or (h >= 7 and hp10 >= 12):
        tg: TopicGranularity = "fine"
    elif h <= 4 and n > 7000:
        tg = "coarse"
    else:
        tg = "medium"

    if math_p10 > 11.0 or proof_p10 > 7.0:
        fd: FormalDensity = "high"
    elif math_p10 < 2.5 and proof_p10 < 2.0:
        fd = "low"
    else:
        fd = "medium"

    if def_p10 > 22.0:
        cd: ConceptualDensity = "high"
    elif def_p10 < 7.0:
        cd = "low"
    else:
        cd = "medium"

    return sgs, tg, fd, cd


def _practical_exercise_signals(sample: str) -> tuple[bool, PracticalDensity, bool]:
    """
    Detect exercise sheets / task-heavy combined sources.
    Returns (has_exercise_material, practical_density, problem_solving_emphasis).
    """
    has_marker = bool(_EXERCISE_SHEET_MARKERS.search(sample))
    task_score = sum(float(len(p.findall(sample))) for p in _TASK_LANGUAGE)
    n = max(len(sample), 1)
    per_10k = task_score / (n / 10_000.0)

    has_exercise_material = has_marker or per_10k >= 11.0

    if per_10k < 5.0 and not has_marker:
        dens: PracticalDensity = "low"
    elif per_10k > 19.0 or has_marker:
        dens = "high"
    else:
        dens = "medium"

    pse = bool(has_exercise_material or per_10k >= 7.0)
    return has_exercise_material, dens, pse


def analyze_extracted_text(text: str) -> LectureAnalysis:
    """
    Analyze truncated or full extracted lecture text.
    """
    sample = text if len(text) <= 120_000 else text[:120_000]
    lang = _detect_language(sample)
    ms = _math_score(sample)
    cs = _code_score(sample)
    profile, hf, hc = _pick_profile(ms, cs)

    org_hits = _pattern_hits(_ORG_PATTERNS, sample)
    proof_hits = _pattern_hits(_PROOF_PATTERNS, sample)
    def_hits = _pattern_hits(_DEF_PATTERNS, sample)
    ex_hits = _pattern_hits(_EXAMPLE_PATTERNS, sample)
    heading_lines = len(_HEADING_LINE.findall(sample))

    kind, is_org, is_proof = _classify_lecture_kind(
        profile=profile,
        math_s=ms,
        code_s=cs,
        org_hits=org_hits,
        proof_hits=proof_hits,
        def_hits=def_hits,
        ex_hits=ex_hits,
        heading_lines=heading_lines,
        n_chars=len(sample),
    )
    depth = _depth_band(sample, ms, heading_lines)
    has_ex, pract_dens, pse = _practical_exercise_signals(sample)
    sgs, tgran, fden, cden = _structural_signals(
        sample, math_s=ms, proof_hits=proof_hits, def_hits=def_hits, heading_lines=heading_lines
    )

    notes = (
        f"heuristic math_score={ms:.1f} code_score={cs:.1f} org={org_hits:.1f} proof={proof_hits:.1f} "
        f"def={def_hits:.1f} ex={ex_hits:.1f} headings={heading_lines} kind={kind} depth={depth} "
        f"exercise_material={has_ex} practical={pract_dens} pse={pse} "
        f"grounding={sgs} topic_gran={tgran} formal={fden} conceptual={cden}"
    )
    return LectureAnalysis(
        detected_language=lang,
        content_profile=profile,
        has_formulas=hf,
        has_code=hc,
        notes=notes,
        lecture_kind=kind,
        depth_band=depth,
        is_organizational=is_org,
        is_proof_heavy=is_proof,
        has_exercise_material=has_ex,
        practical_density=pract_dens,
        problem_solving_emphasis=pse,
        source_grounding_strength=sgs,
        topic_granularity=tgran,
        formal_density=fden,
        conceptual_density=cden,
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
        kind = block.get("lecture_kind", "general")
        if kind not in _VALID_KINDS:
            kind = "general"
        depth = block.get("depth_band", "medium")
        if depth not in _VALID_DEPTH:
            depth = "medium"
        pd = block.get("practical_density", "medium")
        if pd not in ("low", "medium", "high"):
            pd = "medium"
        sgs = block.get("source_grounding_strength", "medium")
        if sgs not in _VALID_GROUND:
            sgs = "medium"
        tgr = block.get("topic_granularity", "medium")
        if tgr not in _VALID_GRANULARITY:
            tgr = "medium"
        fd = block.get("formal_density", "medium")
        if fd not in _VALID_GROUND:
            fd = "medium"
        cd = block.get("conceptual_density", "medium")
        if cd not in _VALID_GROUND:
            cd = "medium"
        return LectureAnalysis(
            detected_language=lang,
            content_profile=prof,
            has_formulas=bool(block.get("has_formulas")),
            has_code=bool(block.get("has_code")),
            notes=str(block.get("notes") or ""),
            lecture_kind=kind,  # type: ignore[arg-type]
            depth_band=depth,  # type: ignore[arg-type]
            is_organizational=bool(block.get("is_organizational", kind == "organizational")),
            is_proof_heavy=bool(block.get("is_proof_heavy", kind == "proof_heavy")),
            has_exercise_material=bool(block.get("has_exercise_material")),
            practical_density=pd,  # type: ignore[arg-type]
            problem_solving_emphasis=bool(block.get("problem_solving_emphasis")),
            source_grounding_strength=sgs,  # type: ignore[arg-type]
            topic_granularity=tgr,  # type: ignore[arg-type]
            formal_density=fd,  # type: ignore[arg-type]
            conceptual_density=cd,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        return None
