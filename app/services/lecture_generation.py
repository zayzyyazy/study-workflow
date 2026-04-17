"""Generate study materials (Markdown) into outputs/ and artifacts table."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from app.services import artifact_service, lecture_service, openai_service
from app.services.generation_readiness import prepare_generation_inputs
from app.services import lecture_meta
from app.services.generation_markdown_cleanup import cleanup_generated_markdown
from app.services.lecture_analysis import LectureAnalysis, analyze_extracted_text
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.study_output_paths import build_study_pack_markdown

# Avoid huge prompts: model context is limited; this keeps requests safe for typical lectures.
MAX_LECTURE_CHARS = 120_000
# Cap how much of the topic map we pass into the core_learning prompt as context
_TOPIC_MAP_CONTEXT_CHARS = 3_000


def _truncate_for_generation(text: str) -> str:
    if len(text) <= MAX_LECTURE_CHARS:
        return text
    return (
        text[:MAX_LECTURE_CHARS]
        + "\n\n---\n\n*[Note: Lecture text was truncated for generation because it exceeded "
        + str(MAX_LECTURE_CHARS)
        + " characters. Re-run after splitting the source if you need full coverage.]*\n"
    )


def _base_user_block(course_name: str, lecture_title: str, lecture_text: str) -> str:
    return (
        f"Course: {course_name}\n"
        f"Lecture title: {lecture_title}\n\n"
        f"Lecture content (Markdown/plain text):\n\n{lecture_text}"
    )


def _run_one(
    *,
    system: str,
    extra_user_instruction: str,
    course_name: str,
    lecture_title: str,
    lecture_text: str,
    max_tokens: int = 4096,
) -> tuple[bool, str, str]:
    user = extra_user_instruction.strip() + "\n\n" + _base_user_block(course_name, lecture_title, lecture_text)
    return openai_service.chat_completion_markdown(
        system_prompt=system, user_prompt=user, max_tokens=max_tokens
    )


def _wants_math_rules(a: LectureAnalysis) -> bool:
    return a.content_profile in ("math", "mixed") or a.has_formulas


def _wants_code_rules(a: LectureAnalysis) -> bool:
    return a.content_profile in ("code", "mixed") or a.has_code


def _profile_rules(a: LectureAnalysis) -> str:
    """Extra system rules for math/code/mixed; language-aware; tight formatting for Markdown + KaTeX."""
    if a.detected_language == "de":
        parts: list[str] = []
        if _wants_math_rules(a):
            parts.append(
                "Mathematik — Formatierung (verbindlich):\n"
                "- Inline-Formeln immer in einfachen Dollarzeichen: $...$ (nicht nur Kursiv/Unterstriche).\n"
                "- Größere oder abgesetzte Formeln/Definitionen in $$...$$ auf eigenen Zeilen; vor/nach ggf. Leerzeile.\n"
                "- Indizes und „Unterstriche" nur innerhalb von $...$ (z. B. $x_1$, $a_{ij}$), nie rohe _ außerhalb, sonst zerstört Markdown die Darstellung.\n"
                "- Symbole und Bezeichner möglichst exakt wie in der Quelle; keine neuen Gleichungen erfinden.\n"
                "- Lieber klare Notation als vage Umschreibung in Prosa, wenn die Vorlesung formal arbeitet.\n"
                "- Nach zentralen Formeln kurz auf Deutsch erklären, was die Symbole bedeuten."
            )
        if _wants_code_rules(a):
            parts.append(
                "Code — Formatierung (verbindlich):\n"
                "- Code in fenced Markdown-Blöcken (dreifache Backticks); nach den Backticks wenn möglich eine Sprache nennen (z. B. ```python, ```text).\n"
                "- Einrückung und Zeilenumbrüche aus der Quelle bewahren; Code nicht in einen Satz „hineinquetschen".\n"
                "- Funktionsnamen, Variablen, APIs und Schlüsselwörter unverändert lassen; nicht still „reparieren" oder umbenennen.\n"
                "- Wenn du vereinfachtes oder künstliches Beispielcode zeigst, klar als Beispiel kennzeichnen.\n"
                "- Verständliche Erklärung auf Deutsch neben oder unter dem Block, was der Code tut.\n"
                "- Erklärungen auf Deutsch; Bezeichner, APIs und Schlüsselwörter im Code exakt wie in der Quelle (oft Englisch) — nicht „übersetzen"."
            )
        if a.content_profile == "mixed" and _wants_math_rules(a) and _wants_code_rules(a):
            parts.append(
                "Gemischter Inhalt: Mathe mit $...$ / $$...$$ und Code mit fenced Blocks strikt trennen; erklärender Text auf Deutsch dazwischen. "
                "Eine Regel darf die andere nicht zerstören (keine Kursiv-Interpretation statt Formel)."
            )
        if not parts:
            parts.append(
                "Fachbegriffe aus der Quelle beibehalten, wenn sie dort üblich sind; nicht unnötig vereinfachen."
            )
        return "\n\n".join(parts)
    parts_en: list[str] = []
    if _wants_math_rules(a):
        parts_en.append(
            "Mathematics — formatting (mandatory):\n"
            "- Inline math: always wrap in single-dollar delimiters: $...$ (do not rely on raw underscores outside math).\n"
            "- Standalone or prominent equations/definitions: use $$...$$ on their own lines; add blank lines around when appropriate.\n"
            "- Subscripts/superscripts only inside $...$ (e.g. $x_1$, $a^{2}$) so Markdown does not eat underscores or italics.\n"
            "- Preserve symbols and notation from the source; do not invent new equations or identities.\n"
            "- Prefer faithful notation over vague paraphrase when the lecture is formal.\n"
            "- After key formulas, add a short plain-language sentence explaining variables or meaning."
        )
    if _wants_code_rules(a):
        parts_en.append(
            "Code — formatting (mandatory):\n"
            "- Use fenced Markdown code blocks (triple backticks); add a language tag when clear (```python, ```bash, ```text).\n"
            "- Preserve indentation and line breaks from the source when possible; do not flatten code into prose.\n"
            "- Keep function names, variables, APIs, and keywords exactly as in the source; do not silently rewrite or fix them.\n"
            "- If you show a simplified or illustrative snippet, label it explicitly as an example.\n"
            "- Explain behavior in plain language after or beside the block.\n"
            "- Prose follows the lecture language; keep code identifiers, APIs, and keywords exactly as in the source (often English) — do not translate them."
        )
    if a.content_profile == "mixed" and _wants_math_rules(a) and _wants_code_rules(a):
        parts_en.append(
            "Mixed technical content: apply both math rules ($...$ / $$...$$) and fenced code blocks in the same output; "
            "keep them visually separate. Explanatory prose should not break formulas or code identifiers."
        )
    if not parts_en:
        parts_en.append(
            "Keep standard technical terms from the source accurate; avoid oversimplifying vocabulary."
        )
    return "\n\n".join(parts_en)


def _artifact_technical_addon(a: LectureAnalysis, step: str) -> str:
    """
    Short, task-specific reminders (user message) for math/code/mixed — keeps language of prose intact.
    """
    wm = _wants_math_rules(a)
    wc = _wants_code_rules(a)
    if not wm and not wc:
        return ""

    if a.detected_language == "de":
        bullets: list[str] = []
        if step == "topic_map" and wm:
            bullets.append(
                "- Topic Map: Begriffe mit mathematischen Symbolen in `$...$` setzen; bei Bedarf `$$...$$` für längere Ausdrücke. "
                "Keine losen Unterstriche für Indizes."
            )
        if step == "topic_map" and wc:
            bullets.append(
                "- Topic Map: Code-Identifier und Schlüsselwörter in `Backticks`; mehrzeilige Signaturen in fenced Blocks."
            )
        if step == "core_learning" and wm:
            bullets.append(
                "- Lernen/Erklären: Formeln in `$...$` / `$$...$$`; pro Hauptthema klar mit ##/### gliedern. "
                "Bei eingebauten Beispielen: jeden Rechenschritt mit `$...$` zeigen."
            )
        if step == "core_learning" and wc:
            bullets.append("- Lernen/Erklären: Code in fenced Blocks zeigen, wenn die Vorlesung Code nutzt.")
        if step == "revision_sheet" and wm:
            bullets.append(
                "- Merkblatt: Kernaussagen und Regeln in `$...$`; nur zum Auswendigen, was die Vorlesung wirklich verlangt."
            )
        if step == "revision_sheet" and wc:
            bullets.append("- Merkblatt: kurze Code-Snippets nur wenn prüfungsrelevant, in fenced Blocks.")
        if not bullets:
            return ""
        return "\n\nZusätzlich für diese Ausgabe:\n" + "\n".join(bullets)

    bullets_en: list[str] = []
    if step == "topic_map" and wm:
        bullets_en.append(
            "- Topic Map: put mathematical symbols/expressions in `$...$`; use `$$...$$` for larger expressions. "
            "Avoid raw underscores for subscripts outside math mode."
        )
    if step == "topic_map" and wc:
        bullets_en.append(
            "- Topic Map: put code identifiers/keywords in `backticks`; use fenced blocks for multi-line signatures."
        )
    if step == "core_learning" and wm:
        bullets_en.append(
            "- Core Learning: use `$...$` / `$$...$$` for formulas; structure each main topic with ##/### headings. "
            "For inline examples: show each calculation step in `$...$`."
        )
    if step == "core_learning" and wc:
        bullets_en.append("- Core Learning: use fenced code blocks when the lecture uses code.")
    if step == "revision_sheet" and wm:
        bullets_en.append(
            "- Revision sheet: compact formulas/rules in `$...$`; separate memorize vs understand."
        )
    if step == "revision_sheet" and wc:
        bullets_en.append("- Revision sheet: tiny code snippets only if exam-relevant; fenced blocks.")
    if not bullets_en:
        return ""
    return "\n\nAdditional requirements for this output:\n" + "\n".join(bullets_en)


def _system_prompt(a: LectureAnalysis) -> str:
    if a.detected_language == "de":
        base = (
            "Du bist ein Lern-Coach / Tutor: Ziel ist echtes Üben und Verstehen — nicht nur eine kurze Zusammenfassung. "
            "Erkläre verständlich, strukturiere für Wiederholung und Prüfung, nenne typische Fehler und Checks. "
            "Schreibe in Markdown mit Überschriften (##, ###), Listen und **Fettdruck** für zentrale Begriffe. "
            "Nur an der Vorlesung orientiert; nichts erfinden. "
            "Die Sprach-/Inhaltsanalyse unten steuert die Formatierung; der Vorlesungstext ist maßgeblich."
        )
        analysis = (
            f"Voranalyse: Ausgaben durchgehend auf **Deutsch**. "
            f"Inhaltstyp: {a.content_profile}. "
            f"Quelle enthält erkennbare Formeln: {'ja' if a.has_formulas else 'nein'}. "
            f"Quelle enthält erkennbaren Code: {'ja' if a.has_code else 'nein'}."
        )
    else:
        base = (
            "You are a learning coach / tutor: prioritize real teaching — not a thin abstract summary. "
            "Explain clearly, structure for review and exams, surface common mistakes and self-checks. "
            "Use Markdown with headings (##, ###), lists, and bold for key terms. "
            "Be accurate to the lecture only; do not invent facts. "
            "The lecture analysis below guides formatting; the lecture text itself is authoritative."
        )
        analysis = (
            f"Lecture analysis: write **all** outputs in **English**. "
            f"Content profile: {a.content_profile}. "
            f"Source appears to contain formulas: {'yes' if a.has_formulas else 'no'}. "
            f"Source appears to contain code: {'yes' if a.has_code else 'no'}."
        )
    return base + "\n\n" + analysis + "\n\n" + _profile_rules(a)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _prompt_quick_overview(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Quick Overview** — kurze Orientierung, die ich vor allem anderen lese.\n\n"
            "Beantworte klar und kompakt:\n"
            "- Worum geht es in dieser Vorlesung?\n"
            "- Was ist die zentrale Idee / das Kernthema?\n"
            "- Warum ist das wichtig — was bringt mir das Verständnis davon?\n"
            "- Wo passt diese Vorlesung in den Kurs — was kam davor, was ermöglicht sie?\n"
            "- Worauf soll ich mich gedanklich einstellen, bevor ich die Details lese?\n\n"
            "Format:\n"
            "- 5–8 knappe Bullet-Points oder kurze Absätze.\n"
            "- Kein Fülltext. Kein generisches Lehrbuch-Intro.\n"
            "- Keine langen Definitionen, keine ausgearbeiteten Beispiele — die kommen später.\n"
            "- Bezug auf den tatsächlichen Vorlesungsinhalt; nichts Erfundenes.\n\n"
            "Oberste Überschrift exakt: ## Quick Overview"
        )
    else:
        extra = (
            "Produce **Quick Overview** — the short orientation read before everything else.\n\n"
            "Answer clearly and concisely:\n"
            "- What is this lecture about?\n"
            "- What is the central idea / main theme?\n"
            "- Why does it matter — what does understanding this enable?\n"
            "- Where does it fit in the course — what came before, what does this set up?\n"
            "- What should I mentally focus on before reading the detailed materials?\n\n"
            "Format:\n"
            "- 5–8 tight bullets or short paragraphs.\n"
            "- No generic filler. No textbook-style preamble.\n"
            "- No full definitions, no worked examples — those come later.\n"
            "- Ground every statement in the actual lecture, not generic claims.\n\n"
            "Top heading must be exactly: ## Quick Overview"
        )
    return sys, extra


def _prompt_topic_map(
    a: LectureAnalysis,
    sibling_titles: list[str] | None = None,
) -> tuple[str, str]:
    sys = _system_prompt(a)

    # Build course context block for cross-lecture connections
    if sibling_titles:
        if a.detected_language == "de":
            course_ctx = (
                "\n\nAndere Vorlesungen in diesem Kurs (für Kurs-Verbindungen):\n"
                + "\n".join(f"- {t}" for t in sibling_titles[:20])
            )
        else:
            course_ctx = (
                "\n\nOther lectures in this course (for cross-lecture connections):\n"
                + "\n".join(f"- {t}" for t in sibling_titles[:20])
            )
    else:
        course_ctx = ""

    if a.detected_language == "de":
        extra = (
            "Erstelle **Topic Map** — die strukturelle Karte dieser Vorlesung.\n\n"
            "Ziel: Zeige, woraus die Vorlesung besteht, was zentral ist und wie die Themen zusammenhängen.\n\n"
            "Wichtig:\n"
            "- Nur die wirklich wichtigen Themen/Konzepte dieser Vorlesung (ca. 6–15).\n"
            "- Nicht jeder Begriff braucht einen Slot — nur Themen mit echter Rolle in der Vorlesung.\n"
            "- Keine langen Erklärungen hier — das ist eine Karte, kein Lernabschnitt.\n\n"
            "Für jedes Thema, dieses Format verwenden:\n\n"
            "### [Themenname]\n"
            "**Tiefe:** X/10\n"
            "**Was es ist:** Kurze, präzise Definition (1–2 Sätze)\n"
            "**Warum wichtig:** Kurze Begründung bezogen auf diese Vorlesung\n"
            "**Verbindungen:** Wie es sich zu anderen Themen in dieser Vorlesung verhält\n"
            "**Kurs-Link:** *(nur wenn sicher ableitbar)* Baut auf früheren Vorlesungsthemen auf / Grundlage für spätere Themen\n\n"
            "---\n\n"
            "Tiefenscore (1–10) basiert auf:\n"
            "- Wie viel Vorlesungstext dem Thema gewidmet ist\n"
            "- Ob es Definitionen/Notation/Formeln hat\n"
            "- Ob Beispiele darauf aufbauen\n"
            "- Ob spätere Ideen es voraussetzen\n"
            "- Ob es wiederholt betont wird\n\n"
            "Kurs-Link-Regeln:\n"
            "- Nur wenn ein echter Bezug zum Kurs klar erkennbar ist.\n"
            "- Vage Allgemeinaussagen weglassen.\n"
            "- Bei Unsicherheit: weglassen statt erfinden.\n\n"
            "Oberste Überschrift exakt: ## Topic Map"
            + course_ctx
            + _artifact_technical_addon(a, "topic_map")
        )
    else:
        extra = (
            "Produce **Topic Map** — the structural map of this lecture.\n\n"
            "Goal: show what this lecture is made of, what is central, and how topics connect.\n\n"
            "Rules:\n"
            "- Only the genuinely important topics/concepts from this lecture (aim for 6–15).\n"
            "- Not every term needs a slot — only topics with a real role in the lecture.\n"
            "- No long explanations here — this is a map, not a teaching section.\n\n"
            "For each topic, use this exact format:\n\n"
            "### [Topic Name]\n"
            "**Depth:** X/10\n"
            "**What it is:** Short precise definition (1–2 sentences)\n"
            "**Why it matters:** Brief reason tied to this lecture\n"
            "**Connections:** How it relates to other topics in this lecture\n"
            "**Course link:** *(only if safely inferable)* Builds on earlier lectures / foundational for later topics\n\n"
            "---\n\n"
            "Depth score (1–10) based on:\n"
            "- How much lecture text is devoted to it\n"
            "- Whether it has formal definitions/notation/formulas\n"
            "- Whether examples depend on it\n"
            "- Whether later ideas in the lecture build on it\n"
            "- Whether it is repeatedly emphasized\n\n"
            "Course link rules:\n"
            "- Only include if a genuine connection to the course is clearly inferable.\n"
            "- Omit vague generic lines like 'important in many fields'.\n"
            "- When uncertain: omit rather than invent.\n\n"
            "Top heading must be exactly: ## Topic Map"
            + course_ctx
            + _artifact_technical_addon(a, "topic_map")
        )
    return sys, extra


def _prompt_core_learning(
    a: LectureAnalysis,
    topic_map_content: str | None = None,
) -> tuple[str, str]:
    """Main teaching file — tutor-style, depth calibrated by topic map scores."""
    sys = _system_prompt(a)

    # Inject topic map as calibration context if available
    if topic_map_content:
        truncated = topic_map_content[:_TOPIC_MAP_CONTEXT_CHARS]
        if a.detected_language == "de":
            map_block = (
                "\n\nDie Topic Map dieser Vorlesung (zur Kalibrierung der Erklärungstiefe):\n\n"
                f"{truncated}\n\n"
                "Nutze die Tiefenscores (1–10) aus der Topic Map:\n"
                "- Tiefe 7–10: ausführlich erklären (Intuition + Formales + Warum + Anwendung + Beispiel + typische Verwechslung).\n"
                "- Tiefe 4–6: klare Erklärung + warum es wichtig ist; evtl. kurzes Beispiel.\n"
                "- Tiefe 1–3: knapp halten — genug zum Verstehen, nicht aufblähen.\n"
            )
        else:
            map_block = (
                "\n\nTopic Map for this lecture (use to calibrate explanation depth):\n\n"
                f"{truncated}\n\n"
                "Use the depth scores (1–10) from the Topic Map:\n"
                "- Depth 7–10: explain thoroughly (intuition + formal + why + usage + example + pitfall).\n"
                "- Depth 4–6: clear explanation + why it matters; maybe a short example.\n"
                "- Depth 1–3: keep brief — enough to understand, not inflated.\n"
            )
    else:
        if a.detected_language == "de":
            map_block = (
                "\n\nAdaptive Tiefe (da keine Topic Map verfügbar): Beurteile selbst anhand von Überschriften, "
                "Wiederholungen und Formalisierungsgrad, was zentral ist, und erkläre dementsprechend tief.\n"
            )
        else:
            map_block = (
                "\n\nAdaptive depth (no Topic Map available): judge from lecture headings, repetition, and "
                "formalization what is central, and calibrate depth accordingly.\n"
            )

    if a.detected_language == "de":
        extra = (
            "Erstelle **Core Learning** als Haupt-Lernteil.\n"
            "Schreibe wie ein Tutor — erkläre die wichtigen Teile der Vorlesung wirklich, nicht nur auflisten.\n\n"
            "Struktur:\n"
            "- Oberste Überschrift exakt: ## Core Learning\n"
            "- Dann Hauptthemen mit ### Überschriften (je Thema ein Abschnitt).\n"
            "- Unterabschnitte mit #### nur wenn nötig.\n\n"
            "Inhalt pro Thema (je nach Tiefe):\n"
            "- Hohe Tiefe: Intuition in klarer Sprache · Formale Definition/Regel/Notation · Warum es wichtig ist · "
            "Wie man es in Aufgaben erkennt und anwendet · Typische Verwechslung wenn plausibel · Beispiel wenn hilfreich.\n"
            "- Mittlere Tiefe: Klare Erklärung + Warum es wichtig ist · Vielleicht ein kurzes Beispiel.\n"
            "- Geringe Tiefe: Kurze Erklärung — genug zum Verstehen, nicht mehr.\n\n"
            "Strikte Regeln:\n"
            "- Nicht jedes Thema gleich lang behandeln — Tiefe folgt der Vorlesungstiefe.\n"
            "- Keine Glossar-Definitionen in voller Länge wiederholen (Topic Map hat sie bereits).\n"
            "- Keine Quick-Overview-Orientierung wiederholen.\n"
            "- Nur Inhalte aus der Vorlesung; keine erfundene Erweiterung.\n"
            "- Revision-Sheet-Inhalte (reine Stichpunkte, Merklisten) gehören nicht hierher — die kommen später."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
        )
    else:
        extra = (
            "Produce **Core Learning** as the main learning section.\n"
            "Write like a tutor — actually explain the important parts of the lecture, not just list them.\n\n"
            "Structure:\n"
            "- Top heading must be exactly: ## Core Learning\n"
            "- Then main topics as ### headings (one section per topic).\n"
            "- Sub-sections with #### only when needed.\n\n"
            "Per-topic content (scaled by depth):\n"
            "- High depth: intuition in plain language · formal definition/rule/notation · why it matters · "
            "how to recognize and apply it · common confusion if plausible · example if helpful.\n"
            "- Medium depth: clear explanation + why it matters · maybe a short example.\n"
            "- Low depth: brief explanation only — enough to understand, no more.\n\n"
            "Strict rules:\n"
            "- Do NOT give every topic equal depth — depth follows lecture depth.\n"
            "- Do not restate Topic Map definitions in full (they are already there).\n"
            "- Do not repeat Quick Overview orientation.\n"
            "- Stay grounded in the lecture; do not invent a broader curriculum.\n"
            "- Pure bullet-point memorization lists belong in the Revision Sheet, not here."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
        )
    return sys, extra


def _prompt_revision_sheet(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Revision Sheet** — kurze, prüfungsnahe Wiederholungsseite.\n\n"
            "Aufbau:\n"
            "- **Auswendig lernen**: Kernregeln, Formeln, Symbole, Fakten — nur was die Vorlesung wirklich verlangt.\n"
            "- **Konzeptuell verstehen**: Wichtige Ideen, die ich auf Anhieb erklären können muss.\n\n"
            "Regeln:\n"
            "- Kurz, skimmbar, prüfungsorientiert.\n"
            "- Keine langen Erklärungen — die stehen in Core Learning.\n"
            "- Keine neuen Themen einführen.\n"
            "- Bullet-Listen oder kompakte Tabellen bevorzugen.\n\n"
            "Oberste Überschrift exakt: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
        )
    else:
        extra = (
            "Produce a **Revision Sheet** — short, exam-friendly, skimmable.\n\n"
            "Structure:\n"
            "- **Memorize**: key rules, formulas, symbols, facts — only what the lecture requires.\n"
            "- **Understand conceptually**: important ideas I must be able to explain on the spot.\n\n"
            "Rules:\n"
            "- Keep it concise and skimmable.\n"
            "- No long explanations — those belong in Core Learning.\n"
            "- No new material; only what the lecture supports.\n"
            "- Prefer bullet lists or compact tables.\n\n"
            "Top heading must be exactly: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
        )
    return sys, extra


# ---------------------------------------------------------------------------
# Generation steps: (artifact_type, filename, prompt_fn, max_tokens)
# ---------------------------------------------------------------------------

GENERATION_STEPS: list[
    tuple[str, str, Callable[..., tuple[str, str]], int]
] = [
    ("quick_overview", "01_quick_overview.md", _prompt_quick_overview, 3072),
    ("topic_map", "02_topic_map.md", _prompt_topic_map, 4096),
    ("core_learning", "03_core_learning.md", _prompt_core_learning, 8192),
    ("revision_sheet", "04_revision_sheet.md", _prompt_revision_sheet, 6144),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sibling_titles(lecture_id: int, course_id: int) -> list[str]:
    """Return titles of other lectures in the same course (for cross-lecture context)."""
    try:
        lectures = lecture_service.list_lectures_for_course(course_id)
        return [lec["title"] for lec in lectures if int(lec["id"]) != lecture_id]
    except Exception:  # noqa: BLE001
        return []


def _sync_meta(
    lecture_root: Path,
    lec: dict[str, Any],
    *,
    status: str,
    extraction_message: Optional[str] = None,
    generation_message: Optional[str] = None,
    generated_artifacts: Optional[list[dict[str, str]]] = None,
    lecture_analysis: Optional[dict[str, Any]] = None,
) -> None:
    prev = lecture_meta.read_meta(lecture_root)
    payload = lecture_meta.build_meta_payload(
        lecture_id=int(lec["id"]),
        course_name=lec["course_name"],
        lecture_name=lec["title"],
        source_file_name=lec["source_file_name"],
        source_file_path=lec["source_file_path"],
        extracted_text_path=lec.get("extracted_text_path"),
        status=status,
        created_at=str(lec["created_at"]),
        extraction_message=extraction_message if extraction_message is not None else prev.get("extraction_message"),
        generation_message=generation_message,
        generated_artifacts=generated_artifacts,
        lecture_analysis=lecture_analysis,
        previous=prev,
    )
    lecture_meta.write_meta(lecture_root, payload)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_study_materials_generation(lecture_id: int) -> tuple[bool, str]:
    """
    Full pipeline: readiness → analysis → generation_pending → OpenAI calls → files + DB → generation_complete,
    or generation_failed on error. Missing API key returns (False, message) without changing status.
    """
    if not openai_service.is_openai_configured():
        return False, "OpenAI is not configured. Set OPENAI_API_KEY in your .env file (see README)."

    prep = prepare_generation_inputs(lecture_id)
    if not prep.ok or not prep.payload:
        return False, prep.reason

    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    root = lecture_root_from_source_relative(lec["source_file_path"])
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    course_name = prep.payload["course_name"]
    lecture_title = prep.payload["lecture_title"]
    lecture_text = _truncate_for_generation(prep.payload["extracted_text"])
    analysis = analyze_extracted_text(lecture_text)
    analysis_meta = analysis.to_meta_dict()

    # Sibling lecture titles for cross-lecture context in Topic Map
    sibling_titles = _get_sibling_titles(lecture_id, int(lec["course_id"]))

    lecture_service.update_lecture_status(lecture_id, "generation_pending")
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if lec:
        _sync_meta(
            root,
            lec,
            status="generation_pending",
            generation_message="Generation started.",
            lecture_analysis=analysis_meta,
        )

    results: list[tuple[str, str]] = []
    written_paths: list[Path] = []
    topic_map_md: str | None = None  # passed to core_learning as calibration context

    for artifact_type, filename, prompt_fn, max_tok in GENERATION_STEPS:
        # Build prompt — pass extra context for topic_map and core_learning
        if artifact_type == "topic_map":
            system, user_extra = prompt_fn(analysis, sibling_titles=sibling_titles)
        elif artifact_type == "core_learning":
            system, user_extra = prompt_fn(analysis, topic_map_content=topic_map_md)
        else:
            system, user_extra = prompt_fn(analysis)

        ok, md, err = _run_one(
            system=system,
            extra_user_instruction=user_extra,
            course_name=course_name,
            lecture_title=lecture_title,
            lecture_text=lecture_text,
            max_tokens=max_tok,
        )
        if not ok:
            for p in written_paths:
                try:
                    p.unlink()
                except OSError:
                    pass
            lecture_service.update_lecture_status(lecture_id, "generation_failed")
            lec2 = lecture_service.get_lecture_by_id(lecture_id)
            if lec2:
                _sync_meta(
                    root,
                    lec2,
                    status="generation_failed",
                    generation_message=err or "Generation failed.",
                    lecture_analysis=analysis_meta,
                )
            return False, err or "Generation failed."

        out_path = outputs_dir / filename
        md = cleanup_generated_markdown(md)
        out_path.write_text(md + "\n", encoding="utf-8")
        written_paths.append(out_path)
        rel = lecture_meta.relative_to_app(out_path)
        results.append((artifact_type, rel))

        # Save topic map content so core_learning can use it
        if artifact_type == "topic_map":
            topic_map_md = md

    pack_body = cleanup_generated_markdown(build_study_pack_markdown(outputs_dir))
    pack_path = outputs_dir / "05_study_pack.md"
    pack_path.write_text(pack_body + "\n", encoding="utf-8")
    written_paths.append(pack_path)
    results.append(("study_pack", lecture_meta.relative_to_app(pack_path)))

    artifact_service.replace_generation_artifacts(lecture_id, results)

    lecture_service.update_lecture_status(lecture_id, "generation_complete")
    lec3 = lecture_service.get_lecture_by_id(lecture_id)
    if not lec3:
        return True, "Generation finished but lecture record could not be reloaded."

    gen_list = [{"artifact_type": t, "file_path": p} for t, p in results]
    _sync_meta(
        root,
        lec3,
        status="generation_complete",
        generation_message="Study materials generated successfully.",
        generated_artifacts=gen_list,
        lecture_analysis=analysis_meta,
    )
    msg = "Study materials generated successfully."
    try:
        from app.services.course_concept_index import index_lecture_safe

        idx_warn = index_lecture_safe(lecture_id)
        if idx_warn:
            msg = f"{msg} Concept indexing note: {idx_warn}"
    except Exception as e:  # noqa: BLE001
        msg = f"{msg} Concept indexing note: {e}"
    return True, msg
