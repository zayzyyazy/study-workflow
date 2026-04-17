"""Generate study materials (Markdown) into outputs/ and artifacts table."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import GENERATION_MODE
from app.services import artifact_service, lecture_service, openai_service
from app.services.generation_readiness import prepare_generation_inputs
from app.services import lecture_meta
from app.services.generation_markdown_cleanup import cleanup_generated_markdown
from app.services.lecture_analysis import LectureAnalysis, analyze_extracted_text
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.source_manifest import split_combined_extracted_text
from app.services.study_output_paths import build_study_pack_markdown

# Avoid huge prompts: model context is limited; this keeps requests safe for typical lectures.
MAX_LECTURE_CHARS = 120_000
# After role split, cap layers so exercise sheets do not crowd out lecture core.
MAX_LAYER_LECTURE_CORE_CHARS = 92_000
MAX_LAYER_EXERCISE_CHARS = 24_000
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


def _truncate_layered_lecture_exercise(lecture_core: str, exercise_text: str) -> tuple[str, str]:
    """Per-role caps so Übungsmaterial stays secondary in the prompt budget."""
    note = "\n\n---\n\n*[Note: Truncated for generation.]*\n"
    ex = exercise_text
    if len(ex) > MAX_LAYER_EXERCISE_CHARS:
        ex = ex[:MAX_LAYER_EXERCISE_CHARS] + note
    if not ex.strip():
        lc = lecture_core
        if len(lc) > MAX_LECTURE_CHARS:
            lc = lc[:MAX_LECTURE_CHARS] + note
        return lc, ""
    lc = lecture_core
    if len(lc) > MAX_LAYER_LECTURE_CORE_CHARS:
        lc = lc[:MAX_LAYER_LECTURE_CORE_CHARS] + note
    return lc, ex


def _layered_material_block(
    course_name: str,
    lecture_title: str,
    lecture_core: str,
    exercise_text: str,
    *,
    language_is_de: bool,
    is_organizational: bool,
) -> str:
    """
    Primary block = Vorlesung (and notes/other); secondary = Übungs-PDFs.
    Admin/logistics must not drive structure unless the unit is organizational (flag).
    """
    org_note_de = ""
    org_note_en = ""
    if is_organizational:
        org_note_de = (
            "\n(Hinweis: Diese Einheit wirkt überwiegend organisatorisch — Logistik darf hier sachlich vorkommen.)\n"
        )
        org_note_en = (
            "\n(Note: This unit looks mostly organizational — logistics may appear proportionally.)\n"
        )

    if language_is_de:
        primary = (
            f"Course: {course_name}\n"
            f"Lecture title: {lecture_title}\n"
            f"{org_note_de}"
            f"\n### Vorlesungskern (primär)\n"
            f"Nutze diesen Block für **Quick Overview**, **Roadmap/Inhaltsverzeichnis**, **Topic Roadmap** und "
            f"**Topic Lessons**: echte Überschriften, Begriffe und Unterkapitel aus der Vorlesung.\n"
            f"Ignoriere **Organisatorisches, Moodle, Termine, Übungsgruppenwahl, Installationshinweise, "
            f"„Nächste Schritte“** als **keine** zentralen Lernthemen — außer die Einheit ist klar nur Logistik.\n\n"
            f"{lecture_core}"
        )
        secondary = (
            f"\n\n---\n\n### Übungs- / Aufgabenmaterial (sekundär)\n"
            f"Nur für: typische **Fragestellungen**, **Aufgabenmuster**, **Missverständnisse**, "
            f"**praktische Relevanz**, **Prüfungsnähe** — **nicht** für die Themenstruktur, **nicht** als Ersatz "
            f"für Vorlesungsüberschriften, **keine** künstlichen „Übungs-Themen“ in der Roadmap.\n\n"
            f"{exercise_text}"
        )
    else:
        primary = (
            f"Course: {course_name}\n"
            f"Lecture title: {lecture_title}\n"
            f"{org_note_en}"
            f"\n### Primary lecture core\n"
            f"Use this block for **Quick Overview**, **roadmap/TOC**, **Topic Roadmap**, and **Topic Lessons**: "
            f"real headings, terms, and subsections from the lecture.\n"
            f"Treat **logistics, LMS, deadlines, exercise-group signup, install reminders, “next steps”** as "
            f"**not** central study topics — unless the unit is clearly administrative only.\n\n"
            f"{lecture_core}"
        )
        secondary = (
            f"\n\n---\n\n### Secondary: exercises / practice sheets\n"
            f"Use only for: typical **task patterns**, **wording**, **misconceptions**, **practical emphasis**, "
            f"**exam relevance** — **do not** replace lecture structure; **do not** invent roadmap topics from tasks.\n\n"
            f"{exercise_text}"
        )
    return primary + secondary


def _material_user_block(
    course_name: str,
    lecture_title: str,
    lecture_core: str,
    exercise_text: str,
    *,
    language_is_de: bool,
    is_organizational: bool,
) -> str:
    if not (exercise_text or "").strip():
        return _base_user_block(course_name, lecture_title, lecture_core)
    return _layered_material_block(
        course_name,
        lecture_title,
        lecture_core,
        exercise_text.strip(),
        language_is_de=language_is_de,
        is_organizational=is_organizational,
    )


def _run_one(
    *,
    system: str,
    extra_user_instruction: str,
    course_name: str,
    lecture_title: str,
    material_block: str,
    max_tokens: int = 4096,
) -> tuple[bool, str, str]:
    user = extra_user_instruction.strip() + "\n\n" + material_block
    return openai_service.chat_completion_markdown(
        system_prompt=system, user_prompt=user, max_tokens=max_tokens
    )


def _wants_math_rules(a: LectureAnalysis) -> bool:
    if a.is_organizational:
        return False
    return a.content_profile in ("math", "mixed") or a.has_formulas


def _wants_code_rules(a: LectureAnalysis) -> bool:
    if a.is_organizational:
        return False
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
                "- Indizes und „Unterstriche“ nur innerhalb von $...$ (z. B. $x_1$, $a_{ij}$), nie rohe _ außerhalb, sonst zerstört Markdown die Darstellung.\n"
                "- Symbole und Bezeichner möglichst exakt wie in der Quelle; keine neuen Gleichungen erfinden.\n"
                "- Lieber klare Notation als vage Umschreibung in Prosa, wenn die Vorlesung formal arbeitet.\n"
                "- Nach zentralen Formeln kurz auf Deutsch erklären, was die Symbole bedeuten."
            )
        if _wants_code_rules(a):
            parts.append(
                "Code — Formatierung (verbindlich):\n"
                "- Code in fenced Markdown-Blöcken (dreifache Backticks); nach den Backticks wenn möglich eine Sprache nennen (z. B. ```python, ```text).\n"
                "- Einrückung und Zeilenumbrüche aus der Quelle bewahren; Code nicht in einen Satz „hineinquetschen“.\n"
                "- Funktionsnamen, Variablen, APIs und Schlüsselwörter unverändert lassen; nicht still „reparieren“ oder umbenennen.\n"
                "- Wenn du vereinfachtes oder künstliches Beispielcode zeigst, klar als Beispiel kennzeichnen.\n"
                "- Verständliche Erklärung auf Deutsch neben oder unter dem Block, was der Code tut.\n"
                "- Erklärungen auf Deutsch; Bezeichner, APIs und Schlüsselwörter im Code exakt wie in der Quelle (oft Englisch) — nicht „übersetzen“."
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
                "- Topic-Lektionen (Mathe): Formeln in `$...$` / `$$...$$` in den passenden Lektions-Abschnitten. "
                "Verknüpfe Formeln mit **erklärender Prosa** — keine Ketten aus Einzeiler-Bullets. "
                "Rechenbeispiele: Schritte **zusammenhängend** und Ausdrücke in `$...$`."
            )
        if step == "core_learning" and wc:
            bullets.append(
                "- Topic-Lektionen (Code): fenced Blocks wenn die Vorlesung Code nutzt; **vor/nach** dem Block kurz "
                "**was** der Code tut — in der jeweiligen Lektion."
            )
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
            "- Topic Lessons (math): use `$...$` / `$$...$$` inside the relevant lesson sections. "
            "Weave formulas into **explanatory prose** — avoid chains of one-line bullets. "
            "For worked steps: **connected sentences** and expressions in `$...$`."
        )
    if step == "core_learning" and wc:
        bullets_en.append(
            "- Topic Lessons (code): fenced blocks when the lecture uses code; **before/after** each block, briefly "
            "say **what** it does — within that lesson."
        )
    if step == "revision_sheet" and wm:
        bullets_en.append(
            "- Revision sheet: compact formulas/rules in `$...$`; separate memorize vs understand."
        )
    if step == "revision_sheet" and wc:
        bullets_en.append("- Revision sheet: tiny code snippets only if exam-relevant; fenced blocks.")
    if not bullets_en:
        return ""
    return "\n\nAdditional requirements for this output:\n" + "\n".join(bullets_en)


def _analysis_signal_lines(a: LectureAnalysis) -> str:
    """Extra deterministic cues for the system prompt (both languages)."""
    if a.detected_language == "de":
        g = a.source_grounding_strength
        tg = a.topic_granularity
        fd = a.formal_density
        cd = a.conceptual_density
        pd = a.practical_density
        g_lab = {"low": "dünne/kurze Quelle — Ausgabe bewusst eng halten", "medium": "mittel", "high": "umfangreiche/strukturierte Quelle — nah an Definitionen/Beispielen bleiben"}
        tg_lab = {
            "coarse": "wenig Zwischenüberschriften — keine künstliche Feinaufteilung erzwingen",
            "medium": "mittlere Struktur",
            "fine": "viele Überschritten/Blöcke — präzise Unterthemen statt breiter Schirmbegriffe",
        }
        fd_lab = {"low": "wenig Formalismus", "medium": "mittlere Formalität", "high": "stark formal/beweis-/symbolnah"}
        cd_lab = {"low": "weniger Definitions-/Distinktionsdichte", "medium": "mittlere Begriffsdichte", "high": "definitions- und unterscheidungsreich"}
        return (
            f"Heuristiken (nur Steuerung, nichts erfinden): **Quellen-Verankerung** — {g_lab.get(g, g)}; "
            f"**Themen-Granularität** — {tg_lab.get(tg, tg)}; **Formalitätsgrad** — {fd_lab.get(fd, fd)}; "
            f"**Begriffsdichte** — {cd_lab.get(cd, cd)}; **Übungs-/Aufgabenanteil** — **{pd}**."
        )
    g = a.source_grounding_strength
    tg = a.topic_granularity
    fd = a.formal_density
    cd = a.conceptual_density
    pd = a.practical_density
    g_lab = {
        "low": "thin/short source — keep outputs narrow",
        "medium": "medium",
        "high": "long/structured source — stay close to definitions and examples",
    }
    tg_lab = {
        "coarse": "few sub-headings — do not force fake fine structure",
        "medium": "medium structure",
        "fine": "many titled blocks — prefer precise subtopics over umbrella labels",
    }
    fd_lab = {"low": "low formalism", "medium": "medium formality", "high": "formal / proof- / symbol-heavy"}
    cd_lab = {
        "low": "lower definition/distinction density",
        "medium": "medium conceptual density",
        "high": "definition- and distinction-rich",
    }
    return (
        f"Heuristics (steering only; invent nothing): **source anchoring** — {g_lab.get(g, g)}; "
        f"**topic granularity** — {tg_lab.get(tg, tg)}; **formality** — {fd_lab.get(fd, fd)}; "
        f"**conceptual density** — {cd_lab.get(cd, cd)}; **exercise/task density** — **{pd}**."
    )


def _strict_source_faithfulness(a: LectureAnalysis) -> str:
    """Cross-cutting exam-prep / professor discipline (system prompt)."""
    if a.detected_language == "de":
        thin = ""
        if a.source_grounding_strength == "low":
            thin = (
                "\n**Dünne Quelle:** Wenn wenig Text vorliegt, bleibt die Ausgabe **kurz und ehrlich schmal** — "
                "**keine** Schein-Tiefe, keine ergänzenden Standardkapitel aus Allgemeinwissen.\n"
            )
        return (
            "**Rolle: strenge:r Hochschul-Studienpartner:in — nur aus dem Material.**\n"
            "- Nutze **ausschließlich** den Vorlesungstext und angehängte Übungs-/Tutoriumsdateien. "
            "Kein Ausweichen auf allgemeines Lehrbuchwissen, wenn es **nicht** in der Quelle vorkommt oder sich nicht "
            "sicher daraus ergibt.\n"
            "- **Kein** „Feld erklären“, **kein** Blog-Artikel, **kein** motivierender Warmton — **sachlich, direkt, präzise**.\n"
            "- **Prüfungs-/Lernnutzen** über **präzise, quellgestützte** Unterscheidungen und Regeln — nicht über breite Zusammenfassung.\n"
            "- **Vollständigkeit** ist kein Ziel: **Nutzen** schlägt Umfang. Lieber **weniger** Text mit klaren Kanten.\n"
            "- Kursweite Verbindungen nur, wenn die **Quelle** oder der **Kurskontext** sie trägt — keine Spekulation.\n"
            "- Übungsmaterial **schärft** Relevanz und Fragetypen, **ersetzt** aber keine Definitionen aus der Vorlesung.\n"
            + thin
        )
    thin_en = ""
    if a.source_grounding_strength == "low":
        thin_en = (
            "\n**Thin source:** If little text is available, stay **short and honestly narrow** — "
            "no fake depth, no generic textbook padding.\n"
        )
    return (
        "**Role: strict university study partner — from the uploaded material only.**\n"
        "- Use **only** the lecture text and attached exercise/tutorium sources. "
        "Do **not** substitute general textbook knowledge unless it is **clearly** in the source or safely implied there.\n"
        "- **No** discipline overview, **no** blog tone, **no** warm motivational voice — **neutral, direct, precise**.\n"
        "- Exam prep value comes from **sharp, source-backed** distinctions and rules — not from broad summarization.\n"
        "- **Completeness is not a goal:** **usefulness** beats length. Prefer **less** text with clear edges.\n"
        "- Course-wide links only when the **source** or **course context** supports them — no speculation.\n"
        "- Exercise sheets **sharpen** relevance and question types; they **do not** replace lecture definitions.\n"
        + thin_en
    )


def _adaptation_summary(a: LectureAnalysis) -> str:
    """Human-readable heuristic summary for the system prompt (both languages)."""
    kind = a.lecture_kind
    depth = a.depth_band
    if a.detected_language == "de":
        kind_labels = {
            "organizational": "überwiegend organisatorisch/administrativ",
            "conceptual": "konzeptuell theoretisch",
            "mathematical": "mathematisch",
            "proof_heavy": "beweisorientiert",
            "coding": "programmier-/code-lastig",
            "mixed": "gemischt (Mathe + Code/Technik)",
            "general": "allgemein / nicht eindeutig zugeordnet",
        }
        depth_labels = {"light": "eher leicht/einführend", "medium": "mittlere Dichte", "dense": "dicht/forgeschritten"}
        lines = [
            f"Klassifikation (Heuristik): **{kind_labels.get(kind, kind)}**.",
            f"Geschätzte Vorlesungs-Tiefe: **{depth_labels.get(depth, depth)}**.",
        ]
        if a.is_proof_heavy:
            lines.append("Es gibt starke Signale für **Beweis-/Argumentationsanteile**.")
        if a.is_organizational:
            lines.append(
                "Diese Einheit wirkt **logistik- oder regelzentriert** — keine künstliche Theorie/Mathe simulieren."
            )
        if a.has_exercise_material or a.practical_density == "high":
            lines.append(
                "Es gibt Signale für **Übungs-/Aufgabenmaterial** — Themen aus Vorlesung **und** Übung stärker verknüpfen."
            )
        lines.append(_analysis_signal_lines(a))
        return "\n".join(lines)
    kind_labels = {
        "organizational": "mostly organizational / admin",
        "conceptual": "conceptual / theory-oriented",
        "mathematical": "mathematical",
        "proof_heavy": "proof-heavy",
        "coding": "programming / code-heavy",
        "mixed": "mixed (math + code/technical)",
        "general": "general / not strongly classified",
    }
    depth_labels = {"light": "lighter / introductory", "medium": "medium density", "dense": "dense / advanced"}
    lines = [
        f"Classification (heuristic): **{kind_labels.get(kind, kind)}**.",
        f"Estimated lecture depth: **{depth_labels.get(depth, depth)}**.",
    ]
    if a.is_proof_heavy:
        lines.append("Strong signals for **proof-style reasoning**.")
    if a.is_organizational:
        lines.append("This session looks **logistics- and rules-centric** — do not fabricate theory/math.")
    if a.has_exercise_material or a.practical_density == "high":
        lines.append(
            "Signals for **exercise / worksheet material** — connect lecture topics with **practice patterns** more strongly."
        )
    lines.append(_analysis_signal_lines(a))
    return "\n".join(lines)


def _anti_generic_rules(a: LectureAnalysis) -> str:
    """Cross-cutting rules to reduce uniform, filler-heavy outputs."""
    if a.detected_language == "de":
        return (
            "**Qualitätsvertrag (verbindlich):**\n"
            "**1) Kein generischer Fülltext:** Sätze wie „wichtig in vielen Bereichen“, „hilft beim Verständnis“, "
            "„relevant für weiteres Studium“, „es ist wesentlich zu wissen“ — **nur** wenn die **Vorlesung** das "
            "konkret hergibt. **Test:** Könnte der Satz fast unverändert in eine andere Vorlesung? → **Streichen oder schärfen.**\n"
            "**2) Keine gleiche Gewichtung:** Nicht jedes Thema verdient gleich viel Raum — **wenige zentrale Konzepte** "
            "bekommen den Großteil der Erklärung; Randthemen **kurz**.\n"
            "**3) Kein Fachbuch über das ganze Feld:** Diese **eine Vorlesung** lehren — nicht das gesamte Gebiet.\n"
            "**4) Keine aufgesetzte Praxis:** Keine flachen Real-Life-Beispiele nur für Show. Abstrakt bleiben, wenn die Quelle "
            "abstrakt ist; **Übungs-/Quellbeispiele** vorziehen.\n"
            "**5) Keine Wiederholung zwischen Abschnitten:** Quick Overview ≠ Themen-Roadmap ≠ Topic-Lektionen. "
            "Jeder Abschnitt hat **einen Job** — nicht dieselben Inhalte in anderem Satzbau.\n"
            "**6) Kürzer statt weicher:** Lieber **präzise und selektiv** als lang und höflich. Keine künstliche Vollständigkeit.\n"
            "**7) Prüfungsnähe vor Breite:** Lieber **exakte, quellgestützte Unterscheidungen** als glatt polierte Überblicke.\n"
            "**8) Kein „Belohnen“ von Vollständigkeit:** Lieber Lücken lassen als mit Allgemeinwissen auffüllen.\n"
            "**9) Kein Füller für akademischen Schein:** Keine leeren Satzschleifen, die nur seriös klingen.\n"
            "**10) Kein Überblicks-/Summary-Modus in Topic-Lektionen:** Wenn es wie eine **glatte Zusammenfassung** klingt "
            "statt wie **konkretes Arbeiten mit der Vorlesung**, umschreiben oder streichen.\n"
            "Zusätzlich: Glossar-Definitionen **nicht** in jedem Abschnitt wiederholen; **einmal klar, dann vernetzen**. "
            "Gleich lange, glatt polierte Absätze für jedes Thema vermeiden — **Priorität sichtbar machen**."
        )
    return (
        "**Quality contract (mandatory):**\n"
        "**1) No generic filler:** Ban vague lines like “important in many fields”, “helps students understand”, "
        "“relevant for further study”, “it is essential to know” **unless** the **lecture text** grounds them. "
        "**Test:** Could the sentence fit almost any lecture unchanged? → **Cut or sharpen with lecture-specific grounding.**\n"
        "**2) No equal treatment:** Not every topic deserves equal space — spend most explanation on **a few central ideas**; "
        "keep supporting topics **short**.\n"
        "**3) No broad textbook chapter:** Teach **this lecture**, not the whole discipline.\n"
        "**4) No fake practicality:** Do not invent shallow “real-life” examples for color. If the lecture is abstract, stay "
        "honest; prefer **exercise/source-grounded** illustrations.\n"
        "**5) No repetition across sections:** Quick Overview must not pre-summarize Topic Roadmap / Topic Lessons. "
        "Each section has a **distinct job** — not the same points rephrased.\n"
        "**6) Prefer dense over polite:** **Selective and sharp** beats long and safe. No filler to sound complete.\n"
        "**7) Exam usefulness over breadth:** Prefer **exact, source-backed distinctions** over polished overviews.\n"
        "**8) Do not reward completeness:** Better to leave gaps than pad with general knowledge.\n"
        "**9) No filler for academic tone:** No empty sentences that only sound serious.\n"
        "**10) No overview/summary voice in Topic Lessons:** If it reads like a **smooth recap** instead of **working the "
        "lecture material**, rewrite or cut.\n"
        "Also: **define once**, then reuse; avoid equally polished paragraphs for every topic — **show priority**."
    )


def _scope_and_topic_rules(a: LectureAnalysis) -> str:
    """Outputs must feel specific to this course/lecture, not interchangeable summaries."""
    if a.detected_language == "de":
        return (
            "Bezug & Originalität (verbindlich):\n"
            "- Formulierungen und Schwerpunkte müssen **erkennbar zu genau dieser Vorlesung** passen — "
            "nicht wie ein generisches Fachbuch oder ein „freundlicher Blog“.\n"
            "- **Nur Quelle:** Kein Ausweiten auf das Fachgebiet, keine „gesamter Kurs“-Narrative, wenn die Einheit das nicht trägt.\n"
            "- **Kurs- und Themenabhängigkeit:** Begriffe, Beispiele und Gewichtung aus der **Quelle** ableiten; "
            "keine austauschbare Standard-Erklärung.\n"
            "- **Rhythmus und Schärfe** an die Vorlesung anpassen — nicht jede Einheit gleich lang, gleich weich, gleich „balanced“.\n"
            "- **Abschnitts-Jobs (nicht wiederholen):** Quick Overview = **Kurz-Orientierung + Roadmap-Liste**; "
            "Themen-Roadmap = **Struktur + Tiefe**; Topic-Lektionen = **echtes Lernen pro Thema**; Revision Sheet = **komprimiert** — "
            "keine zweite Erklärphilosophie."
        )
    return (
        "Scope & originality (mandatory):\n"
        "- Wording must **clearly fit this specific lecture** — not a generic textbook or a soft educational blog post.\n"
        "- **Source scope only:** Do not broaden into the whole discipline or invent whole-course narratives the unit does not support.\n"
        "- **Course/topic dependence:** derive terms, examples, and weighting from the **source** only.\n"
        "- **Vary sharpness and length** — not every subsection equally long, equally polite, or equally “balanced”.\n"
        "- **Distinct section jobs (no duplication):** Quick Overview = **short orientation + roadmap list**; "
        "Topic Roadmap = **structure + depth**; Topic Lessons = **real learning per topic**; Revision Sheet = **compressed** — "
        "not a second explainer."
    )


def _exercise_application_addon(a: LectureAnalysis, step: str) -> str:
    """Stronger solving/practice behavior when exercise-like sources or task language are present."""
    if not (a.has_exercise_material or a.problem_solving_emphasis or a.practical_density == "high"):
        return ""
    if a.detected_language == "de":
        base = (
            "\n\n**Übungs-/Anwendungsbezug (Heuristik: zusätzliche Quellen oder Aufgabenanteil erkannt):**\n"
            "- **Vorlesungslogik zuerst** — Übungen **schärfen** Relevanz und Fragetypen, ersetzen aber nicht die Struktur "
            "der Einheit.\n"
            "- Nutze **Aufgaben/Übungsblätter** in der Quelle, um **praktische Gewichtung** und **typische Formulierungen** "
            "zu erkennen.\n"
            "- Themen, die in **Vorlesung und Übung** vorkommen, **stärker gewichten** — nicht nur nennen.\n"
        )
        if step == "topic_map":
            base += (
                "- Topic Map: **praktische Relevanz** (wiederkehrende Übungsmuster) darf **Tiefenscores** und "
                "**Warum wichtig** stärker beeinflussen als reine Randbegriffe.\n"
            )
        elif step == "core_learning":
            base += (
                "- Topic-Lektionen: in **`#### Aufgaben / Prüfungsnähe`** (wenn genutzt) zeigen, wie man Aufgaben angeht — "
                "aus der Quelle; **keine** Meta-Überschriften.\n"
            )
        elif step == "revision_sheet":
            base += (
                "- Revision Sheet: **lösungsnahe** Merkpunkte (Checkliste: typische Aufgabenmuster) wenn die Quelle das hergibt.\n"
            )
        return base
    base = (
        "\n\n**Exercise / application context (heuristic: extra sources or task-heavy text detected):**\n"
        "- **Lecture logic first** — exercises **sharpen** relevance and question types; they do **not** replace lecture structure.\n"
        "- Use worksheets to infer **practical weight** and **typical phrasing**.\n"
        "- Topics in **both lecture and exercises** get **more weight** — not a name-check only.\n"
    )
    if step == "topic_map":
        base += (
            "- Topic Map: **practical relevance** (recurring exercise patterns) may **raise depth scores** and "
            "**why it matters** vs peripheral terms.\n"
        )
    elif step == "core_learning":
        base += (
            "- Topic Lessons: in **`#### Tasks / exam angle`** (when used) show how to approach tasks — "
            "grounded in the source; **no** meta headings.\n"
        )
    elif step == "revision_sheet":
        base += (
            "- Revision Sheet: add **solution-adjacent** memory hooks (typical task patterns) when the source supports it.\n"
        )
    return base


def _example_policy_line(a: LectureAnalysis) -> str:
    """One line on how strongly to use examples (appended where relevant)."""
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return "\nBeispiele: **kaum bis keine**; keine erfundenen Szenarien — nur wenn die Quelle welche hat."
        if k == "conceptual":
            return "\nBeispiele: **sparsam, aber sinnvoll** — nur zur Klärung von Begriffen/Beziehungen."
        if k == "mathematical":
            return "\nBeispiele: **wichtig** — bevorzugt **rechnerisch/symbolisch** oder kleine Mini-Instanzen aus der Vorlesung."
        if k == "proof_heavy":
            return "\nBeispiele: **Argumentations-/Anwendungsbeispiele** (z. B. wie ein Satz greift), nicht nur Wiederholung der Formulierung."
        if k == "coding":
            return "\nBeispiele: **Code-getrieben** — kurze Snippets, Ein-/Ausgabe, typische Fehler."
        if k == "mixed":
            return "\nBeispiele: **ausgewogen** — Mathe und Code getrennt halten, keinen Modus „gewinnen“ lassen."
        return "\nBeispiele: **nur wenn** sie die Vorlesung wirklich verständlicher machen."
    if k == "organizational":
        return "\nExamples: **few or none**; no invented scenarios — only if the source already has them."
    if k == "conceptual":
        return "\nExamples: **sparing but meaningful** — clarify concepts/relationships, not filler."
    if k == "mathematical":
        return "\nExamples: **important** — prefer worked symbolic / small instances grounded in the lecture."
    if k == "proof_heavy":
        return "\nExamples: **reasoning / application** (how a theorem is used), not just restating the statement."
    if k == "coding":
        return "\nExamples: **code-driven** — short snippets, behavior, pitfalls."
    if k == "mixed":
        return "\nExamples: **balanced** — keep math and code separate; do not let one mode dominate."
    return "\nExamples: **only when** they genuinely improve understanding of the lecture."


def _topic_map_depth_calibration(a: LectureAnalysis) -> str:
    """Makes Topic Map scores more meaningful (spread vs compressed)."""
    d = a.depth_band
    if a.detected_language == "de":
        if d == "light":
            return (
                "\nTiefenscores nutzen: **volle Spanne 1–10** nutzen; bei kurzer/leichter Vorlesung oft **1–3** für Randthemen, "
                "**4–7** für Mittelfeld — keine künstliche Inflation."
            )
        if d == "dense":
            return (
                "\nTiefenscores nutzen: **volle Spanne 1–10**; bei dichter Vorlesung sind **8–10** für wirklich zentrale "
                "Trägerideen üblich, **1–3** für echte Randnotizen."
            )
        return (
            "\nTiefenscores nutzen: **volle Spanne 1–10**; verteile **nicht** alles um 5–6 — "
            "orientiere dich an **Wiederholung, Formalisierung, spätere Abhängigkeiten**."
        )
    if d == "light":
        return (
            "\nDepth scores: use the **full 1–10 range**; for lighter/shorter lectures, side topics are often **1–3**, "
            "core ideas **4–7** — do not inflate."
        )
    if d == "dense":
        return (
            "\nDepth scores: use the **full 1–10 range**; dense lectures often warrant **8–10** for true backbone ideas "
            "and **1–3** for real asides."
        )
    return (
        "\nDepth scores: use the **full 1–10 range**; avoid clustering everything around **5–6** — "
        "use **repetition, formalization, and downstream dependencies**."
    )


def _topic_map_granularity_hint(a: LectureAnalysis) -> str:
    """Steer umbrella vs precise topic labels from heuristics."""
    tg = a.topic_granularity
    if a.detected_language == "de":
        if tg == "fine":
            return (
                "\n**Präzision:** Die Quelle wirkt **in Unterkapitel gegliedert** — Topic-Map-Namen **konkret** wählen "
                "(prüfungsrelevante **Untereinheiten**, typische Verwechslungen), **nicht** nur große Schirmbegriffe."
            )
        if tg == "coarse":
            return (
                "\n**Präzision:** Wenig Zwischenüberschriften — **Themen aus dem Text** ableiten, "
                "**keine** künstliche Feinzerlegung erfinden; trotzdem **präzise benennen**, was die Quelle wirklich macht."
            )
        return (
            "\n**Präzision:** Lieber **prüfungsrelevante Konzept-Einheiten** als breite Marketing-Outline-Labels."
        )
    if tg == "fine":
        return (
            "\n**Precision:** The source looks **finely structured** — prefer **concrete** topic names "
            "(exam-relevant **sub-units**, typical confusions), not only umbrella labels."
        )
    if tg == "coarse":
        return (
            "\n**Precision:** Few sub-headings — derive topics **from the text**, do not invent fake fine structure; "
            "still name **precisely** what the source actually does."
        )
    return "\n**Precision:** Prefer **exam-relevant concept units** over broad brochure-style labels."


def _topic_map_kind_focus(a: LectureAnalysis) -> str:
    """What the Topic Map should emphasize for this lecture kind."""
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return (
                "\nSchwerpunkt Topic Map: **Entscheidungen, Regeln, Abläufe, Termine, Erwartungen** — "
                "nicht vorgeben, es gäbe tiefe Fachkonzepte, wenn die Quelle nur organisatorisch ist."
            )
        if k == "conceptual":
            return (
                "\nSchwerpunkt Topic Map: **Begriffsnetz, Unterscheidungen, Ideenhierarchie** — "
                "Formeln nur wenn die Quelle sie wirklich zentral setzt."
            )
        if k == "mathematical":
            return (
                "\nSchwerpunkt Topic Map: **Notation, Definitionen, Objektbeziehungen** — "
                "Symbole und Formeln **exakt** wie in der Quelle."
            )
        if k == "proof_heavy":
            return (
                "\nSchwerpunkt Topic Map: **Sätze/Behauptungen, Annahmen, Beweisideen** — "
                "Verbindungen zwischen „was gesagt wird“ und „wie argumentiert wird“."
            )
        if k == "coding":
            return (
                "\nSchwerpunkt Topic Map: **APIs, Konzepte, Datenfluss, zentrale Code-Ideen** — "
                "Identifier wie in der Quelle."
            )
        if k == "mixed":
            return (
                "\nSchwerpunkt Topic Map: **beide Welten** (Mathe + Code) mit **gleichen Kartenregeln**, "
                "aber klar getrennten Themen."
            )
        return ""
    if k == "organizational":
        return (
            "\nTopic Map focus: **decisions, rules, logistics, deadlines, expectations** — "
            "do not pretend there are deep technical concepts if the source is administrative."
        )
    if k == "conceptual":
        return (
            "\nTopic Map focus: **concept web, distinctions, idea hierarchy** — "
            "formulas only when the lecture truly centers them."
        )
    if k == "mathematical":
        return (
            "\nTopic Map focus: **notation, definitions, relationships** — "
            "symbols and formulas **exactly** as in the source."
        )
    if k == "proof_heavy":
        return (
            "\nTopic Map focus: **claims, assumptions, proof ideas** — "
            "links between what is stated and how it is argued."
        )
    if k == "coding":
        return (
            "\nTopic Map focus: **APIs, concepts, data flow, core code ideas** — "
            "identifiers as in the source."
        )
    if k == "mixed":
        return (
            "\nTopic Map focus: **both** math and code with the **same map discipline**, "
            "but clearly separated topics."
        )
    return ""


def _core_learning_map_depth_calibration(a: LectureAnalysis) -> str:
    """Extra calibration when Topic Map text is injected (depends on depth_band + kind)."""
    d = a.depth_band
    k = a.lecture_kind
    if a.detected_language == "de":
        base = (
            "\nTiefe zusätzlich gewichten nach Vorlesungslage: **Randthemen kurz** (1–3), "
            "**Kernstücke** (7–10) mit **mehr Raum** — wiederholte Begriffe, formale Definitionen und "
            "spätere Abhängigkeiten erhöhen die Tiefe."
        )
        if d == "dense":
            base += (
                "\nDichte Vorlesung: **8–10** nur für wirklich tragende Konzepte; vermeide gleich lange Abschnitte."
            )
        if d == "light":
            base += (
                "\nLeichte Vorlesung: **nicht aufblasen** — niedrige Scores knapp halten, keine künstliche Tiefe."
            )
        if k == "proof_heavy":
            base += (
                "\nBeweislast: bei hohen Scores **Beweisidee, Annahmen, Schlüsselschritte** erklären — "
                "nicht nur Satz formulieren."
            )
        if k == "organizational":
            base += (
                "\nOrganisatorisch: **kurze, praktische** Abschnitte — keine Theorie-/Mathe-Mimikry."
            )
        if k == "conceptual":
            base += (
                "\nKonzeptuell: **Intuition und Unterscheidungen** vor Formalismus, außer die Quelle ist formal."
            )
        base += (
            "\nTiefe zeigen durch **Ausführlichkeit der Erklärung, Übergänge und Vernetzung** — "
            "nicht durch längere Aufzählungslisten."
        )
        return base
    base = (
        "\nFurther depth weighting: **side topics brief** (1–3), **core topics** (7–10) get **more space** — "
        "repetition, formal definitions, and downstream use increase depth."
    )
    if d == "dense":
        base += "\nDense lecture: reserve **8–10** for true backbone ideas; avoid equal-length sections."
    if d == "light":
        base += "\nLight lecture: **do not inflate** low scores — keep side topics short."
    if k == "proof_heavy":
        base += (
            "\nProof-heavy: for high scores explain **proof idea, assumptions, key steps** — "
            "not restating the theorem only."
        )
    if k == "organizational":
        base += "\nOrganizational: **short practical** sections — no fake theory/math voice."
    if k == "conceptual":
        base += "\nConceptual: prioritize **intuition and distinctions** over formalism unless the source is formal."
    base += (
        "\nShow depth through **richer explanation, transitions, and linking** — not through longer bullet lists."
    )
    return base


def _quick_overview_kind_addon(a: LectureAnalysis) -> str:
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return (
                "\n\nSchwerpunkt dieser Quick Overview: **Session-Zweck, Logistik, Termine, Erwartungen, nächste Schritte** — "
                "nicht eine künstliche „zentrale mathematische Idee“ erfinden."
            )
        if k == "proof_heavy":
            return (
                "\n\nErwähne: **welche Hauptresultate/Behauptungen** und **welche Beweisstile** die Vorlesung prägt (kurz)."
            )
        if k == "coding":
            return (
                "\n\nErwähne: **welche Programmierziele** und **welche Artefakte** (Code, APIs) zentral sind."
            )
        return ""
    if k == "organizational":
        return (
            "\n\nEmphasize **session purpose, logistics, deadlines, expectations, next steps** — "
            "do not invent a fake central mathematical idea."
        )
    if k == "proof_heavy":
        return (
            "\n\nBriefly surface **main results/claims** and **what proof style** dominates (short)."
        )
    if k == "coding":
        return (
            "\n\nBriefly surface **programming goals** and **central artifacts** (code/APIs)."
        )
    return ""


def _core_learning_structure_addon(a: LectureAnalysis) -> str:
    """Kind-specific teaching emphasis — internal; must not become visible section titles."""
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return (
                "\n\n**Ton (nur umsetzen, nicht als Überschrift ausgeben):** **praktisch, knapp** — Abläufe, Regeln, was zu tun ist; "
                "keine Theorie-/Mathe-Show."
            )
        if k == "mathematical":
            return (
                "\n\n**Ton (nur umsetzen):** **Notation exakt**, Definitionen, Symbolbedeutung, "
                "Rechenbeispiele wenn die Vorlesung sie führt; typische Verwechslungen. "
                "**Fluss in Prosa** zwischen Definition, Notation und Folgerung; Bullets nur für Fälle/Regeln/Schritte, "
                "wenn die Struktur es verlangt."
            )
        if k == "conceptual":
            return (
                "\n\n**Ton (nur umsetzen):** **Intuition, Begriffsnetz, Zusammenhänge** — "
                "Formeln nur wenn die Quelle sie wirklich braucht. **Keine** breite Philosophie oder Feldgeschichte erfinden — "
                "nur was die Quelle trägt. **Absätze dominieren; Bullets nur punktuell.**"
            )
        if k == "proof_heavy":
            return (
                "\n\n**Ton (nur umsetzen):** **Satz/Behauptung verstehen**, Annahmen, **Beweisidee**, "
                "warum Schritte nötig sind — nicht nur Satz wiederholen."
            )
        if k == "coding":
            return (
                "\n\n**Ton (nur umsetzen):** **Code lesen, Bedeutung, Verhalten**, typische Fehler; "
                "Code in fenced Blocks. **Prosa verbindet** vor/nach Code — keine Bullet-Zeilen statt Erklärung."
            )
        if k == "mixed":
            return (
                "\n\n**Ton (nur umsetzen):** **Balance** — Mathe und Code getrennt halten, "
                "Anteil der Erklärung an den tatsächlichen Quellenanteilen ausrichten."
            )
        return ""
    if k == "organizational":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **practical and short** — flows, rules, what to do; "
            "no fake theory/math performance."
        )
    if k == "mathematical":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **exact notation**, definitions, symbol meaning, "
            "worked steps when the lecture does; typical confusions. **Prose flow** between definition, notation, "
            "and consequences; bullets only for cases/rules/steps when structure demands it."
        )
    if k == "conceptual":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **intuition, concept web, relationships** — "
            "formulas only when the source truly needs them. **No** broad philosophy or field history unless the source does — "
            "**Paragraphs first; bullets only where they help.**"
        )
    if k == "proof_heavy":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **claims, assumptions, proof idea**, why steps matter — "
            "not restating theorems only."
        )
    if k == "coding":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **code meaning, behavior, pitfalls**; fenced code blocks. "
            "**Prose bridges** before/after code — not bullet lines instead of explanation."
        )
    if k == "mixed":
        return (
            "\n\n**Emphasis (weave in; do not print as a heading):** **balance** — keep math and code separate; "
            "match explanation volume to actual source emphasis."
        )
    return ""


def _topic_lessons_prose_instructions(a: LectureAnalysis) -> str:
    """
    Topic Lessons: few units, **varying** internal shape so every lesson does not read like the same AI mold.
    """
    if a.detected_language == "de":
        org = ""
        if a.is_organizational or a.lecture_kind == "organizational":
            org = (
                "\n\n**Organisatorische Vorlesung:** **knapp** — wenige Lektionen, keine Schein-Tiefe, keine Theorie erfinden."
            )
        return (
            "**Topic-Lektionen — Aufbau (verbindlich):**\n"
            "- **Kein Summary-Modus:** keine Einleitungen wie „Diese Vorlesung behandelt…“; **direkt** mit Substanz starten.\n"
            "- Oberste Überschrift exakt: `## Topic-Lektionen`\n"
            "- **Kein** Gesamt-Essay über die ganze Vorlesung — Erklärung in **Lektionen** entlang der **Quellenreihenfolge**.\n"
            "- **3–6** Lektionen mit `### [Titel]` — Titel **aus Vorlesung** (Überschriften / nummerierte Folienzeilen), "
            "**keine** generischen Kapitelnamen.\n"
            "- **Anti-Schablone (wichtig):** **Nicht** in jeder Lektion dieselben `####`-Unterpunkte wiederholen — das wirkt "
            "generisch. Pro Lektion **2–4** Unterabschnitte mit `####`; **Form pro Lektion wechseln** (siehe Archetypen unten).\n"
            "- **Eigene passende `####`-Titel** wählen (an den **Inhalt** gekoppelt), z. B. "
            "`#### Begriffe und Unterscheidungen`, `#### Was die Folien hier betonen`, `#### Kurzbeispiel aus der Vorlesung`, "
            "`#### Wo Übungsaufgaben anknüpfen` — **nicht** immer dieselbe Reihenfolge und **nicht** immer dieselben vier "
            "Überschriften.\n"
            "- **Archetypen (rotieren, Lektion 1→A, 2→B, 3→C, dann wieder):**\n"
            "  - **A — Begriffe:** Schwerpunkt Definitionen/Unterscheidungen; optional ein Mini-Beispiel aus der Quelle.\n"
            "  - **B — Verdichtung:** Schwerpunkt **zusammenhängender Erklärfluss** (weniger Unterüberschriften), was diese "
            "Einheit an diesem Block **wirklich tut**.\n"
            "  - **C — Anwendung/Fallen:** Schwerpunkt typische Verwechslungen oder Aufgabenlogik **nur** wenn die Quelle "
            "dafür Anker bietet.\n"
            "- **Ton:** sachlich, **quellenfixiert** — Sätze, die in **jede** Vorlesung passen, streichen.\n"
            "- **Priorität:** Jede `###`-Lektion an die **Priorität** aus der Roadmap koppeln — **9–10** ausführlich, **4–7** "
            "knapp, **1–3** minimal (**keine** gleiche Länge für alle).\n"
            "- Themen-Roadmap / Quick Overview **nicht** wiederholen.\n\n"
            "**Verboten:** identische `####`-Ketten in jeder Lektion; gleich lange Blöcke; Meta-Labels wie „Gewichtung“.\n"
            + org
        )
    org_en = ""
    if a.is_organizational or a.lecture_kind == "organizational":
        org_en = "\n\n**Organizational lecture:** **short** — few lessons, no fake depth."
    return (
        "**Topic Lessons — structure (mandatory):**\n"
        "- **No summary mode:** no “This lecture covers…” intros; **start with substance**.\n"
        "- Top heading must be exactly: `## Topic Lessons`\n"
        "- **No** single essay for the whole lecture — teach in **lessons** following **source order**.\n"
        "- **3–6** lessons with `### [Title]` — titles from **lecture headings / outline lines**, not generic chapter names.\n"
        "- **Anti-template:** Do **not** repeat the **same** `####` subsection titles in **every** lesson. "
        "Use **2–4** `####` sections per lesson and **vary the shape** lesson-to-lesson (see archetypes).\n"
        "- Pick **content-specific** `####` titles (e.g. `#### Terms and contrasts`, `#### What the slides emphasize here`, "
        "`#### Small example from the lecture`, `#### How tasks hook in`) — **not** the same four headings each time.\n"
        "- **Archetypes (rotate: lesson 1→A, 2→B, 3→C, repeat):**\n"
        "  - **A — Concepts:** focus definitions/distinctions; optional tiny source-grounded example.\n"
        "  - **B — Compression:** more continuous prose, fewer subheads — what this block **actually does** in-unit.\n"
        "  - **C — Pitfalls / tasks:** emphasize confusions or task logic **only** when the source gives anchors.\n"
        "- **Tone:** source-tight — cut sentences that fit any course.\n"
        "- **Priority:** Tie each `###` lesson to **Priority** from the roadmap — **9–10** expansive, **4–7** tight, "
        "**1–3** minimal (**not** equal length for all).\n"
        "- Do not repeat the Topic Roadmap / Quick Overview.\n\n"
        "**Forbidden:** identical `####` chains in every lesson; equally long polished blocks.\n"
        + org_en
    )


def _revision_kind_addon(a: LectureAnalysis) -> str:
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return (
                "\n\nSchwerpunkt Revision Sheet: **Merken & Tun** — Fristen, Regeln, Anforderungen, "
                "Checklisten; keine künstlichen Formeln."
            )
        if k == "mathematical":
            return (
                "\n\nSchwerpunkt Revision Sheet: **Symbole, Definitionen, Regeln** kompakt; "
                "was man auswendig können muss vs. verstehen."
            )
        if k == "proof_heavy":
            return (
                "\n\nSchwerpunkt Revision Sheet: **Satzschablonen**, zentrale Annahmen, "
                "Beweis-/Lösungsmuster die in Prüfungen vorkommen."
            )
        if k == "coding":
            return (
                "\n\nSchwerpunkt Revision Sheet: **APIs/Syntax**, häufige Fehler, Mini-Snippets nur wenn relevant."
            )
        return ""
    if k == "organizational":
        return (
            "\n\nRevision Sheet focus: **remember & do** — deadlines, rules, requirements, checklists; "
            "no fake formulas."
        )
    if k == "mathematical":
        return (
            "\n\nRevision Sheet focus: **symbols, definitions, rules** compactly; memorize vs understand."
        )
    if k == "proof_heavy":
        return (
            "\n\nRevision Sheet focus: **theorem templates**, key assumptions, proof patterns exam-relevant."
        )
    if k == "coding":
        return (
            "\n\nRevision Sheet focus: **APIs/syntax**, common pitfalls, tiny snippets only if relevant."
        )
    return ""


def _system_prompt(a: LectureAnalysis) -> str:
    if a.detected_language == "de":
        base = (
            "Du bist ein **strenger, universitätsnaher Studienpartner** — **nur** aus dem hochgeladenen Material. "
            "**Kein** freundlicher Überblicks-Generator, **kein** Blog-Ton, **keine** weiche Motivation. "
            "Priorisiere was in **dieser** Einheit wirklich zählt; verschwende keine Worte auf Offensichtliches oder "
            "generisches Fachbuchgelaber. "
            "Zeige **wie Begriffe zusammenhängen** (was muss zuerst sitzen), **typische Fehlvorstellungen** und **wie man denkt** — "
            "natürlich im Text, **ohne** Meta-Überschriften wie „Abhängigkeiten“ oder „Gewichtung“ (besonders wo Übungsmaterial existiert). "
            "Markdown mit ##, ###, sparsam Listen, **Fettdruck** nur für echte Schlüsselbegriffe. "
            "Nur Quellinhalt; nichts erfinden. Analyse unten + Vorlesungstext steuern Schärfe und Format."
        )
        analysis = (
            f"Voranalyse: Ausgaben durchgehend auf **Deutsch**. "
            f"Inhaltstyp (Formel/Code-Signale): {a.content_profile}. "
            f"Lektionsart (Heuristik): **{a.lecture_kind}**. "
            f"Geschätzte Tiefe: **{a.depth_band}**. "
            f"Quelle enthält erkennbare Formeln: {'ja' if a.has_formulas else 'nein'}. "
            f"Quelle enthält erkennbaren Code: {'ja' if a.has_code else 'nein'}. "
            f"Übungs-/Aufgabenmaterial (Heuristik): {'ja' if a.has_exercise_material else 'nein'}; "
            f"praktische Dichte: **{a.practical_density}**; "
            f"Problem-/Lösungsfokus: {'ja' if a.problem_solving_emphasis else 'nein'}.\n\n"
            f"{_adaptation_summary(a)}"
        )
    else:
        base = (
            "You are a **strict university-level study partner** — **only** from the uploaded material. "
            "**Not** a friendly overview generator, **not** a blog voice, **not** warm motivational filler. "
            "Prioritize what matters in **this** unit; do not waste words on obvious points or generic textbook prose. "
            "Show **how ideas build on each other**, **common confusions**, and **how to think** — in natural prose, "
            "**not** with meta-headings like “Dependencies” or “Weighting” (especially when exercise material exists). "
            "Use Markdown ##/###, lists sparingly, **bold** only for real anchors. "
            "Ground everything in the source; invent nothing. The analysis below + lecture text set tone and constraints."
        )
        analysis = (
            f"Lecture analysis: write **all** outputs in **English**. "
            f"Content profile (formula/code signals): {a.content_profile}. "
            f"Lecture kind (heuristic): **{a.lecture_kind}**. "
            f"Estimated depth band: **{a.depth_band}**. "
            f"Source appears to contain formulas: {'yes' if a.has_formulas else 'no'}. "
            f"Source appears to contain code: {'yes' if a.has_code else 'no'}. "
            f"Exercise-like material (heuristic): {'yes' if a.has_exercise_material else 'no'}; "
            f"practical density: **{a.practical_density}**; "
            f"problem-solving emphasis: {'yes' if a.problem_solving_emphasis else 'no'}.\n\n"
            f"{_adaptation_summary(a)}"
        )
    return (
        base
        + "\n\n"
        + analysis
        + "\n\n"
        + _strict_source_faithfulness(a)
        + "\n\n"
        + _anti_generic_rules(a)
        + "\n\n"
        + _scope_and_topic_rules(a)
        + "\n\n"
        + _profile_rules(a)
    )


# ---------------------------------------------------------------------------
# strict_v2: structure-anchored prompts (opt-in via GENERATION_MODE)
# ---------------------------------------------------------------------------


def _extract_heading_outline(lecture_text: str, *, max_lines: int = 96, max_chars: int = 14000) -> str:
    """
    Deterministic structure hints: Markdown # headings plus common slide-PDF patterns
    (numbered titles, short title-case lines) — many lectures lack `#` in extraction.
    """
    out: list[str] = []
    acc = 0
    seen: set[str] = set()

    def _push(s: str) -> bool:
        nonlocal acc
        s = s.strip()
        if not s or len(s) < 6:
            return False
        if len(s) > 400:
            s = s[:397] + "..."
        key = s.casefold()
        if key in seen:
            return False
        seen.add(key)
        out.append(s)
        acc += len(s) + 1
        return acc >= max_chars or len(out) >= max_lines

    for line in lecture_text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if re.match(r"^\s{0,3}#{1,3}\s+\S", line):
            if _push(raw):
                break
            continue
        # "3. Menschliche Wahrnehmung" slide / outline lines
        if re.match(r"^\d{1,2}\.\s+\S", raw) and 10 < len(raw) < 220:
            if _push(raw):
                break
            continue
        # ALL CAPS slide titles (common in extracted decks)
        if (
            " " in raw
            and raw.isupper()
            and 12 <= len(raw) <= 90
            and re.search(r"[A-ZÄÖÜ]", raw)
        ):
            if _push(raw):
                break
    return "\n".join(out)


def _topic_map_strict_v2_block(a: LectureAnalysis, heading_block: str) -> str:
    """Structure-anchored roadmap rules + extracted outline (always on — not tied to GENERATION_MODE)."""
    hb = heading_block.strip()
    if a.detected_language == "de":
        body = (
            "\n\n**Struktur-Anker für die Roadmap:**\n"
            "- **Primäre Anker** sind **tatsächliche Überschriften** der Vorlesung (siehe unten). "
            "**Keine** pro-Thema-Erklärabsätze — nur **Name + Priorität** laut Haupt-Instruktion.\n"
            "- **Lieber 4–8 scharfe** Themen als viele breite Schirmbegriffe.\n"
            "- Übungs-PDFs dürfen **Priorität** leicht erhöhen, **ohne** Text unter dem Thema.\n"
        )
        if hb:
            body += "\n**Extrahierte Überschriften (nur Struktur):**\n\n```text\n" + hb + "\n```\n"
        else:
            body += (
                "\n*Hinweis:* Keine `#`-Überschriften gefunden — Themen trotzdem **aus Absätzen und Betonungen** der Quelle "
                "ableiten, nicht aus Allgemeinwissen.\n"
            )
        return body
    body_en = (
        "\n\n**Structure anchors for the roadmap:**\n"
        "- **Primary anchors** are **actual lecture headings** (below). **No** per-topic explanatory paragraphs — "
        "**name + priority only** as in the main instruction.\n"
        "- Prefer **4–8 sharp** topics over many umbrella labels.\n"
        "- Exercise PDFs may **raise priority** slightly — **no** extra text under the topic.\n"
    )
    if hb:
        body_en += "\n**Extracted headings (structure only):**\n\n```text\n" + hb + "\n```\n"
    else:
        body_en += (
            "\n*Note:* No `#` headings found — still derive topics from **paragraph emphasis** in the source, not from "
            "general knowledge.\n"
        )
    return body_en


def _core_learning_strict_v2_block(a: LectureAnalysis, heading_block: str) -> str:
    """Lesson anchoring to real lecture structure (always on)."""
    hb = heading_block.strip()
    if a.detected_language == "de":
        core = (
            "\n\n**strict_v2 — Topic-Lektionen (zusätzlich verbindlich):**\n"
            "- **Kein Recap-Modus:** lieber **unbequeme Präzision** aus der Quelle als glatte Zusammenfassung.\n"
            "- **Vorlesungsweg:** In der **Reihenfolge** der Quelle erklären (wie die Einheit aufbaut), **nicht** als "
            "Feldüberblick oder alphabetische Themenliste.\n"
            "- **`###`-Lektionen** an **echte Überschriften/Folien** anbinden — **konkrete** Begriffe/Beispiele aus dem Text; "
            "**variierende** `####`-Unterpunkte (nicht dieselbe Schablone in jeder Lektion).\n"
            "- **Keine** glatten Recap-Paragraphen ohne vorlesungsspezifische Kanten.\n"
            "- **Übungen:** Lösungsdenken an **dieselben** Begriffe koppeln wie die Vorlesung.\n"
            "- **Priorität 9–10:** Mehr Absätze, mehr Vernetzung/Fehlkanten **nur** quellnah — **nicht** bei 4–7 dasselbe "
            "Volumen erzwingen.\n"
        )
        if hb:
            core += "\n**Gliederungsanker (Überschriften aus der Quelle):**\n\n```text\n" + hb + "\n```\n"
        else:
            core += (
                "\n*Ohne erkannte `#`-Überschriften:* Reihenfolge aus dem **Textfluss** und **Wiederholungen** rekonstruieren.\n"
            )
        return core
    core_en = (
        "\n\n**strict_v2 — Topic Lessons (additionally mandatory):**\n"
        "- **No recap mode:** prefer **uncomfortable precision** from the source over smooth summary.\n"
        "- **Lecture path:** Explain in **source order** (how the unit builds), **not** as a field survey or sorted topic list.\n"
        "- **`###` lessons** should track **major source blocks** — **concrete** terms/examples from the text; "
        "**vary** `####` subheads (do not reuse the same scaffold every lesson).\n"
        "- **No** polished recap paragraphs without lecture-specific edges.\n"
        "- **Exercises:** tie reasoning to the **same** terms as the lecture.\n"
        "- **Priority 9–10:** More paragraphs and linking/pitfalls **only** when source-grounded — **do not** force the same "
        "volume on 4–7.\n"
    )
    if hb:
        core_en += "\n**Heading anchors from the source:**\n\n```text\n" + hb + "\n```\n"
    else:
        core_en += "\n*No `#` headings detected:* reconstruct order from **flow** and **repetition** in the source.\n"
    return core_en


def _quick_overview_strict_v2_addon(a: LectureAnalysis) -> str:
    if GENERATION_MODE != "strict_v2":
        return ""
    if a.detected_language == "de":
        return "\n\n**strict_v2:** Extrem knapp; jeder Satz **quellenfest** — kein breiter Themenüberblick.\n"
    return "\n\n**strict_v2:** Extremely tight; every sentence **source-grounded** — no broad topic survey.\n"


def _revision_strict_v2_addon(a: LectureAnalysis) -> str:
    if GENERATION_MODE != "strict_v2":
        return ""
    if a.detected_language == "de":
        return (
            "\n\n**strict_v2:** Nur **prüfbare** Stichpunkte aus der Quelle — **keine** erklärenden Mini-Absätze, "
            "**keine** generischen Merksätze.\n"
        )
    return (
        "\n\n**strict_v2:** Only **exam-checkable** bullets from the source — **no** mini-explanations, **no** generic slogans.\n"
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _prompt_quick_overview(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Quick Overview** — **minimale Orientierung** aus der Quelle. "
            "**Kein Lehren** hier (dafür Topic-Lektionen); nur **Was steht an** + **Inhaltsverzeichnis**.\n\n"
            "**Genau zwei Teile:**\n\n"
            "**1) Kurzer Einstieg** — **ein** kurzer Absatz **oder** höchstens **2–4 Zeilen** (nicht mehr):\n"
            "- worum es in **dieser** Einheit geht\n"
            "- was der **zentrale Fokus** ist\n\n"
            "**2) Inhaltsverzeichnis / Roadmap** — **nach** dem Einstieg eine zweite Überschrift:\n"
            "Oberste Überschrift des Blocks exakt: `## Inhaltsverzeichnis`\n"
            "Darunter eine **sehr kurze** Liste (Bullet oder nummeriert) mit **5–8** Stichpunkten: **nur** die "
            "**Hauptthemen**, die ich der Reihe nach durcharbeiten soll — **Namen/Labels**, **noch keine Erklärung**, "
            "**keine** zweite Themen-Roadmap wie in `Themen-Roadmap`.\n\n"
            "**Verbote:** lange Einleitung; breite Zusammenfassung der Vorlesung; generische Wichtigkeit; Wiederholung der "
            "späteren Abschnitte; Fachbuch-Überblick; Definitionen oder ausgearbeitete Beispiele; **kein** angeklebter Schlusssatz "
            "über Übungs-/Prüfungsbezug (dafür gibt es andere Artefakte — Overview = nur Vorlesungskern + Inhaltsverzeichnis).\n\n"
            "Struktur:\n"
            "- Erste Zeile des Dokuments: exakt `## Quick Overview`\n"
            "- Dann Einstieg (kurz)\n"
            "- Dann `## Inhaltsverzeichnis` + Liste\n\n"
            "Jeder Satz muss sich auf **diese** Vorlesung beziehen — sonst streichen."
        )
        extra += (
            _quick_overview_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "quick_overview")
            + _quick_overview_strict_v2_addon(a)
        )
    else:
        extra = (
            "Produce **Quick Overview** — **minimal orientation** from the source. "
            "**No teaching** here (that is Topic Lessons); only **what this unit is** + **outline**.\n\n"
            "**Exactly two parts:**\n\n"
            "**1) Short opening** — **one** short paragraph **or** at most **2–4 lines** (not more):\n"
            "- what **this** unit is about\n"
            "- what the **central focus** is\n\n"
            "**2) Table of contents / roadmap** — after the opening, a second heading:\n"
            "Exact heading: `## Roadmap`\n"
            "Then a **very short** list (bullets or numbered) with **5–8** items: **only** the **main topics** I should work "
            "through in order — **titles only**, **no explanations yet**, **not** a second copy of the Topic Roadmap section.\n\n"
            "**Forbidden:** long intro; broad lecture recap; generic importance; repeating later sections; textbook overview; "
            "definitions or worked examples; **no** tacked-on closing line about worksheets/exams (other files cover that).\n\n"
            "Structure:\n"
            "- First line: exactly `## Quick Overview`\n"
            "- Then opening (short)\n"
            "- Then `## Roadmap` + list\n\n"
            "Every sentence must attach to **this** lecture — otherwise cut."
        )
        extra += (
            _quick_overview_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "quick_overview")
            + _quick_overview_strict_v2_addon(a)
        )
    return sys, extra


def _prompt_topic_map(
    a: LectureAnalysis,
    sibling_titles: list[str] | None = None,
    *,
    lecture_text: str | None = None,
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
            "Erstelle eine **sehr kurze Themen-Roadmap** — **nur Orientierung**, **kein** Lehrtext und **kein** zweites "
            "Skript.\n\n"
            "**Problem das du vermeidest:** Lange Roadmaps mit „Was es ist / Warum wichtig / Verbindungen / Kurs-Link“ sind "
            "**wertlose Zusammenfassungsfläche** — der Studierende lernt dort nicht. **Alles, was wirklich lehrt, gehört "
            "in Topic-Lektionen** (andere Datei).\n\n"
            "**Erlaubt in dieser Datei nur:**\n"
            "- Oberste Überschrift exakt: `## Themen-Roadmap`\n"
            "- **Einleitung:** **optional** — höchstens **ein Satz**, oder **ganz weglassen**, wenn die Themenliste schon "
            "orientiert. **Kein** Mini-Essay, **keine** Übungsdetails.\n"
            "- Dann **4–9** Themenblöcke. **Pro Block exakt zwei Zeilen** — **nichts** darunter:\n"
            "  `### [Themenname]`  \n"
            "  `**Priorität:** X/10`  \n"
            "**Kein** `Hinweis:`-, **kein** Zusatzfeld — Orientierung passiert in **einer Zeile** pro Thema.\n\n"
            "**Strikt verboten** unter jedem Thema: **„Was es ist“**, **„Warum wichtig“**, **„Verbindungen“**, "
            "**„Kurs-Link“**, Definitionen, mehrteilige Absätze, Aufzählungen, **gleich lange** Texte pro Thema.\n\n"
            "**Priorität 1–10** = wie viel Raum Topic-Lektionen später einplanen sollen (hoch = mehr Tiefe dort). "
            "Basiert auf: Anteil Vorlesungstext, Wiederholung, Abhängigkeit späterer Teile, Prüfungsnähe aus Quelle.\n\n"
            "**Anti-Broschüre:** Keine generischen Sammeltitel („Grundlagen“, „Überblick“) ohne vorlesungsspezifischen Kern. "
            "Namen **aus Vorlesungsüberschriften** (siehe extrahierte Zeilen unten), höchstens leicht kürzen.\n"
            + course_ctx
            + _artifact_technical_addon(a, "topic_map")
            + _topic_map_strict_v2_block(a, _extract_heading_outline(lecture_text or ""))
        )
    else:
        extra = (
            "Produce a **very short Topic Roadmap** — **orientation only**, **not** teaching and **not** a second script.\n\n"
            "**Avoid:** Long roadmaps with “What it is / Why it matters / Connections / Course link” — that is **low-value "
            "summary surface**. **Real teaching belongs in Topic Lessons** (another file).\n\n"
            "**Allowed in this file only:**\n"
            "- Top heading exactly: `## Topic Roadmap`\n"
            "- **Intro:** **optional** — **at most one sentence**, or **omit** if the topic list already orients the reader. "
            "**No** mini-essay, **no** exercise detail.\n"
            "- Then **4–9** topic blocks. **Exactly two lines per block** — **nothing** below:\n"
            "  `### [Topic name]`  \n"
            "  `**Priority:** X/10`  \n"
            "**No** `Note:` / extra field — one line per topic for orientation.\n\n"
            "**Strictly forbidden** under each topic: **“What it is”**, **“Why it matters”**, **“Connections”**, "
            "**“Course link”**, definitions, multi-sentence blurbs, **equal-length** filler per topic.\n\n"
            "**Priority 1–10** = how much room Topic Lessons should spend later (high = deeper lessons). Base on: lecture "
            "coverage, repetition, downstream dependence, exam salience from the source.\n\n"
            "**Anti-brochure:** No generic umbrella titles. Prefer **lecture heading strings** (see extracted lines below), "
            "shorten lightly at most.\n"
            + course_ctx
            + _artifact_technical_addon(a, "topic_map")
            + _topic_map_strict_v2_block(a, _extract_heading_outline(lecture_text or ""))
        )
    return sys, extra


def _prompt_core_learning(
    a: LectureAnalysis,
    topic_map_content: str | None = None,
    *,
    lecture_text: str | None = None,
) -> tuple[str, str]:
    """Main teaching file — tutor-style, depth calibrated by topic map scores."""
    sys = _system_prompt(a)

    # Inject topic map as calibration context if available
    if topic_map_content:
        truncated = topic_map_content[:_TOPIC_MAP_CONTEXT_CHARS]
        if a.detected_language == "de":
            map_block = (
                "\n\n**Roadmap = nur Prioritätsliste.** Der Auszug ist **kein** Lehrtext — nur **Themennamen** + "
                "**`Priorität: X/10`**. **Hier** passiert das echte Lehren.\n\n"
                f"{truncated}\n\n"
                "**Prioritäts-Stufen (verbindlich — starke Asymmetrie):**\n"
                "- **9–10:** **Kern-Lektionen** — **deutlich** mehr Umfang als andere: mehrere zusammenhängende Absätze, "
                "echte **Erklärtiefe**. **Quelle:** Vernetzung im Vorlesungsflug (was setzt voraus / wofür braucht man es nachher), "
                "**die** zentrale Unterscheidung oder Definition, die die Folien wirklich tragen, **typische** "
                "Verwechslungskante oder **wie** eine Prüfungs-/Übungsfrage stolpern könnte — **nur** wenn die Quelle "
                "dafür Anker liefert. **Kein** generisches Fachbuchgelaber.\n"
                "- **8:** noch **stark**, aber **etwas knapper** als 9–10 — kein „alles gleich lang“.\n"
                "- **4–7:** **kompakt** — sachlich erklären, **ohne** ausufernde „Warum wichtig“-Breite.\n"
                "- **1–3:** **minimal** — Randthema; wenige Absätze, **keine** künstliche Tiefe.\n\n"
                "**Wichtig:** Nur **Priorität 8–10** bekommen die **extra** Lehr-Schichten (Missverständnis, Prüfungskante, "
                "Kette). **Nicht** jede Lektion verlängern — **nur** die mit hoher Zahl.\n"
            )
        else:
            map_block = (
                "\n\n**Roadmap = priority list only.** The excerpt is **not** teaching — usually **topic names** + "
                "**`Priority: X/10`**. **Teaching happens here.**\n\n"
                f"{truncated}\n\n"
                "**Priority tiers (mandatory — strong asymmetry):**\n"
                "- **9–10:** **Core lessons** — **much** longer than others: multiple connected paragraphs, real depth. "
                "**From the source:** how the idea sits in the lecture chain, **the** central distinction/definition the "
                "slides really emphasize, a **typical confusion edge** or **how a task/exam might trip you** — "
                "**only** when grounded. **No** generic textbook voice.\n"
                "- **8:** still **strong**, but **somewhat shorter** than 9–10.\n"
                "- **4–7:** **compact** — clear, **no** sprawling “why it matters” filler.\n"
                "- **1–3:** **minimal** — few paragraphs; **no** fake depth.\n\n"
                "**Only** priorities **8–10** get the **extra** teaching layers (misconception, exam/task angle, chain). "
                "**Do not** lengthen every lesson—**only** high numbers.\n"
            )
    else:
        if a.detected_language == "de":
            map_block = (
                "\n\nAdaptive Tiefe (da keine Themen-Roadmap verfügbar): Beurteile selbst anhand von Überschriften, "
                "Wiederholungen und Formalisierungsgrad, was zentral ist, und wähle **3–6** Lektionen mit passender Tiefe.\n"
            )
        else:
            map_block = (
                "\n\nAdaptive depth (no Topic Roadmap available): judge from lecture headings, repetition, and "
                "formalization what is central, and pick **3–6** lessons with appropriate depth.\n"
            )

    map_block += _core_learning_map_depth_calibration(a)

    if a.detected_language == "de":
        extra = (
            "**Hauptlehrstück (nicht verhandelbar):** Die Topic-Lektionen sind **der einzige Ort** für echtes **Verstehen** — "
            "**kein** „schöne Zusammenfassung“, **kein** Überblicksmodus. Wenn ein Absatz in **jede** Vorlesung passen "
            "könnte, **streichen**.\n\n"
            "Erstelle **Topic-Lektionen** — **wenige** Lektionen, **maximale** Nutzschicht pro Absatz, **tiefer** statt "
            "breiter.\n\n"
            + _topic_lessons_prose_instructions(a)
            + "\n\n"
            "Priorität aus der Roadmap (pro `###`-Lektion die passende Zahl suchen):\n"
            "- **9–10:** **Kern** — **spürbar** länger und dichter als andere Lektionen; extra Lehr-Schicht (siehe Kasten oben).\n"
            "- **8:** stark, aber **kürzer** als 9–10.\n"
            "- **4–7:** **kompakt**.\n"
            "- **1–3:** **kurz**.\n\n"
            "Strikte Regeln:\n"
            "- **Kein** zusammenhängender Gesamt-Essay über die ganze Vorlesung vor oder statt der Lektionen.\n"
            "- Themen-Roadmap / Quick-Overview-Inhaltsverzeichnis **nicht** wiederholen.\n"
            "- **Nur** Quelle + Übungsmaterial; nichts erfinden.\n"
            "- Merklisten → Revision Sheet, nicht hier."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
            + _core_learning_structure_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "core_learning")
            + _core_learning_strict_v2_block(a, _extract_heading_outline(lecture_text or ""))
        )
    else:
        extra = (
            "**Primary teaching (non-negotiable):** Topic Lessons are **the only place** for real **understanding** — "
            "**not** a polished recap, **not** overview mode. If a paragraph could fit **any** lecture, **cut** it.\n\n"
            "Produce **Topic Lessons** — **few** lessons, **maximum** usefulness per paragraph, **deeper** not broader.\n\n"
            + _topic_lessons_prose_instructions(a)
            + "\n\n"
            "Match each `###` lesson to its **Priority** from the roadmap excerpt above:\n"
            "- **9–10:** **core** — **noticeably** longer/denser than other lessons; extra teaching layers (see box above).\n"
            "- **8:** strong, **shorter** than 9–10.\n"
            "- **4–7:** **compact**.\n"
            "- **1–3:** **short**.\n\n"
            "Strict rules:\n"
            "- **No** single continuous essay about the whole lecture instead of lessons.\n"
            "- Do not repeat the Topic Roadmap / Quick Overview roadmap.\n"
            "- **Only** source + exercise material; invent nothing.\n"
            "- Memorize lists → Revision Sheet, not here."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
            + _core_learning_structure_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "core_learning")
            + _core_learning_strict_v2_block(a, _extract_heading_outline(lecture_text or ""))
        )
    return sys, extra


def _prompt_revision_sheet(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Revision Sheet** — **kompakte** Merk- und **prüfungsnahe** Seite (**keine** zweite Erklärphilosophie).\n\n"
            "Aufbau:\n"
            "- **Auswendig lernen**: nur was die Vorlesung **wirklich** einfordert — Formeln, Symbole, Regeln, Fakten.\n"
            "- **Konzeptuell verstehen**: Ideen, die ich **kurz** erklären können muss — **ohne** ausufernde Prosa.\n\n"
            "Regeln:\n"
            "- **Priorität:** **Unterscheidungen, Definitionen, Regeln, typische Fragetypen/Fehlmuster** — alles **quellgestützt**.\n"
            "- **Selektiv und kurz** — lieber wenige harte Punkte als lange Listen.\n"
            "- **Keine** ausführlichen Erklärungen (→ Topic-Lektionen); nur **Stichworte, Checks, typische Fallen**.\n"
            "- Typische **Frage-/Fehlermuster** und **Unterscheidungen** wenn Übungsmaterial nahelegt.\n"
            "- Keine neuen Themen; **kein** allgemeines Fachwissen ergänzen.\n"
            "- Bullets / Mini-Tabellen; **dichte** Zeilen.\n\n"
            "Oberste Überschrift exakt: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
            + _revision_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "revision_sheet")
            + _revision_strict_v2_addon(a)
        )
    else:
        extra = (
            "Produce a **Revision Sheet** — **compact**, **exam-faithful** cram sheet (**not** a second explainer).\n\n"
            "Structure:\n"
            "- **Memorize**: only what the lecture **actually** demands — rules, formulas, symbols, facts.\n"
            "- **Understand conceptually**: ideas I can explain **briefly** — **no** essay prose.\n\n"
            "Rules:\n"
            "- **Priority:** **distinctions, definitions, rules, typical question types / failure modes** — all **source-backed**.\n"
            "- **Selective and short** — fewer hard hits beat long lists.\n"
            "- **No** long explanations (→ Topic Lessons); **keywords, checks, typical traps** only.\n"
            "- If exercises suggest it: **question types**, **distinctions**, **common mistakes**.\n"
            "- No new material; **do not** pad with general-domain knowledge.\n"
            "- Bullets / compact tables; **tight** lines.\n\n"
            "Top heading must be exactly: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
            + _revision_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "revision_sheet")
            + _revision_strict_v2_addon(a)
        )
    return sys, extra


# ---------------------------------------------------------------------------
# Generation steps: (artifact_type, filename, prompt_fn, max_tokens)
# ---------------------------------------------------------------------------

GENERATION_STEPS: list[
    tuple[str, str, Callable[..., tuple[str, str]], int]
] = [
    ("quick_overview", "01_quick_overview.md", _prompt_quick_overview, 3072),
    # Minimal roadmap — almost only names + priority; Topic Lessons carry teaching.
    ("topic_map", "02_topic_map.md", _prompt_topic_map, 1024),
    ("core_learning", "03_core_learning.md", _prompt_core_learning, 9216),
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
    lecture_core_raw, exercise_raw, _ = split_combined_extracted_text(lecture_text)
    if not (lecture_core_raw or "").strip():
        lecture_core_raw = lecture_text
    lecture_core_t, exercise_t = _truncate_layered_lecture_exercise(lecture_core_raw.strip(), (exercise_raw or "").strip())
    analysis = analyze_extracted_text(
        lecture_text,
        generation_mode=GENERATION_MODE,
        lecture_core_text=lecture_core_t,
        exercise_text=exercise_t,
    )
    material_block = _material_user_block(
        course_name,
        lecture_title,
        lecture_core_t,
        exercise_t,
        language_is_de=analysis.detected_language == "de",
        is_organizational=analysis.is_organizational,
    )
    analysis_meta = analysis.to_meta_dict()
    analysis_meta["generation_mode_used"] = GENERATION_MODE

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
            system, user_extra = prompt_fn(
                analysis, sibling_titles=sibling_titles, lecture_text=lecture_core_t
            )
        elif artifact_type == "core_learning":
            system, user_extra = prompt_fn(
                analysis, topic_map_content=topic_map_md, lecture_text=lecture_core_t
            )
        else:
            system, user_extra = prompt_fn(analysis)

        ok, md, err = _run_one(
            system=system,
            extra_user_instruction=user_extra,
            course_name=course_name,
            lecture_title=lecture_title,
            material_block=material_block,
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
