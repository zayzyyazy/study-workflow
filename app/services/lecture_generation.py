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
                "- Core Learning (Mathe): Formeln in `$...$` / `$$...$$`. "
                "Verknüpfe Formeln mit **erklärender Prosa** — keine Ketten aus Einzeiler-Bullets für jeden Schritt. "
                "Rechenbeispiele: Schritte **zusammenhängend erklären** und Ausdrücke in `$...$` setzen."
            )
        if step == "core_learning" and wc:
            bullets.append(
                "- Core Learning (Code): fenced Blocks wenn die Vorlesung Code nutzt; **vor/nach** dem Block kurz "
                "**was** der Code tut und **wie** er zum restlichen Thema passt — nicht nur nackter Code."
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
            "- Core Learning (math): use `$...$` / `$$...$$` for formulas. "
            "Weave formulas into **explanatory prose** — avoid chains of one-line bullets per micro-step. "
            "For worked steps: explain the **flow in connected sentences** and put expressions in `$...$`."
        )
    if step == "core_learning" and wc:
        bullets_en.append(
            "- Core Learning (code): fenced blocks when the lecture uses code; **before/after** each block, briefly "
            "say **what** it does and **how** it fits the topic — not code alone."
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
            "**5) Keine Wiederholung zwischen Abschnitten:** Quick Overview ≠ Mini-Fassung von Topic Map/Core Learning. "
            "Jeder Abschnitt hat **einen Job** — nicht dieselben Inhalte in anderem Satzbau.\n"
            "**6) Kürzer statt weicher:** Lieber **präzise und selektiv** als lang und höflich. Keine künstliche Vollständigkeit.\n"
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
        "**5) No repetition across sections:** Quick Overview must not pre-summarize Topic Map / Core Learning. "
        "Each section has a **distinct job** — not the same points rephrased.\n"
        "**6) Prefer dense over polite:** **Selective and sharp** beats long and safe. No filler to sound complete.\n"
        "Also: **define once**, then reuse; avoid equally polished paragraphs for every topic — **show priority**."
    )


def _scope_and_topic_rules(a: LectureAnalysis) -> str:
    """Outputs must feel specific to this course/lecture, not interchangeable summaries."""
    if a.detected_language == "de":
        return (
            "Bezug & Originalität (verbindlich):\n"
            "- Formulierungen und Schwerpunkte müssen **erkennbar zu genau dieser Vorlesung** passen — "
            "nicht wie ein generisches Fachbuch oder ein „freundlicher Blog“.\n"
            "- **Kurs- und Themenabhängigkeit:** Begriffe, Beispiele und Gewichtung aus der **Quelle** ableiten; "
            "keine austauschbare Standard-Erklärung.\n"
            "- **Rhythmus und Schärfe** an die Vorlesung anpassen — nicht jede Einheit gleich lang, gleich weich, gleich „balanced“.\n"
            "- **Abschnitts-Jobs (nicht wiederholen):** Quick Overview = **Orientierung**; Topic Map = **selektive Struktur**; "
            "Core Learning = **harte Kernpunkte, Abhängigkeiten, Missverständnisse, Denken**; Revision Sheet = **komprimiert** — "
            "keine zweite Erklärphilosophie."
        )
    return (
        "Scope & originality (mandatory):\n"
        "- Wording must **clearly fit this specific lecture** — not a generic textbook or a soft educational blog post.\n"
        "- **Course/topic dependence:** derive terms, examples, and weighting from the **source** only.\n"
        "- **Vary sharpness and length** — not every subsection equally long, equally polite, or equally “balanced”.\n"
        "- **Distinct section jobs (no duplication):** Quick Overview = **orientation**; Topic Map = **selective structure**; "
        "Core Learning = **dependencies, pitfalls, reasoning**; Revision Sheet = **compressed** — not a second explainer."
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
                "- Core Learning: erkläre **wie man Aufgaben angeht** (Muster, typische ersten Schritte, worauf achten) — "
                "aus der Quelle, nicht erfunden.\n"
                "- **Abhängigkeiten:** was muss zuerst sitzen, was folgt daraus — besonders für Übungsaufgaben.\n"
            )
        elif step == "quick_overview":
            base += (
                "- Quick Overview: erwähne kurz, ob **Übungs-/Prüfungsnahes** dabei ist (ohne Vorweg-Liste aller Aufgaben).\n"
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
            "- Core Learning: explain **how to approach tasks** (patterns, first steps, what to watch for) — "
            "grounded in the source, not invented.\n"
            "- **Dependencies:** what must be solid first, what follows — especially for exercises.\n"
        )
    elif step == "quick_overview":
        base += (
            "- Quick Overview: briefly note if **exam-/practice-adjacent** material is present (no exhaustive task list).\n"
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
    k = a.lecture_kind
    if a.detected_language == "de":
        if k == "organizational":
            return (
                "\n\nSchwerpunkt Core Learning: **praktisch, knapp** — Abläufe, Regeln, was zu tun ist; "
                "keine Theorie-/Mathe-Show."
            )
        if k == "mathematical":
            return (
                "\n\nSchwerpunkt Core Learning: **Notation exakt**, Definitionen, Symbolbedeutung, "
                "Rechenbeispiele wenn die Vorlesung sie führt; typische Verwechslungen. "
                "**Fluss in Prosa** zwischen Definition, Notation und Folgerung; Bullets nur für Fälle/Regeln/Schritte, "
                "wenn die Struktur es verlangt."
            )
        if k == "conceptual":
            return (
                "\n\nSchwerpunkt Core Learning: **Intuition, Begriffsnetz, Zusammenhänge** — "
                "Formeln nur wenn die Quelle sie wirklich braucht. **Absätze dominieren; Bullets nur punktuell.**"
            )
        if k == "proof_heavy":
            return (
                "\n\nSchwerpunkt Core Learning: **Satz/Behauptung verstehen**, Annahmen, **Beweisidee**, "
                "warum Schritte nötig sind — nicht nur Satz wiederholen."
            )
        if k == "coding":
            return (
                "\n\nSchwerpunkt Core Learning: **Code lesen, Bedeutung, Verhalten**, typische Fehler; "
                "Code in fenced Blocks. **Prosa verbindet** vor/nach Code — keine Bullet-Zeilen statt Erklärung."
            )
        if k == "mixed":
            return (
                "\n\nSchwerpunkt Core Learning: **Balance** — Mathe und Code getrennt halten, "
                "Anteil der Erklärung an den tatsächlichen Quellenanteilen ausrichten."
            )
        return ""
    if k == "organizational":
        return (
            "\n\nCore Learning focus: **practical and short** — flows, rules, what to do; "
            "no fake theory/math performance."
        )
    if k == "mathematical":
        return (
            "\n\nCore Learning focus: **exact notation**, definitions, symbol meaning, "
            "worked steps when the lecture does; typical confusions. **Prose flow** between definition, notation, "
            "and consequences; bullets only for cases/rules/steps when structure demands it."
        )
    if k == "conceptual":
        return (
            "\n\nCore Learning focus: **intuition, concept web, relationships** — "
            "formulas only when the source truly needs them. **Paragraphs first; bullets only where they help.**"
        )
    if k == "proof_heavy":
        return (
            "\n\nCore Learning focus: **claims, assumptions, proof idea**, why steps matter — "
            "not restating theorems only."
        )
    if k == "coding":
        return (
            "\n\nCore Learning focus: **code meaning, behavior, pitfalls**; fenced code blocks. "
            "**Prose bridges** before/after code — not bullet lines instead of explanation."
        )
    if k == "mixed":
        return (
            "\n\nCore Learning focus: **balance** — keep math and code separate; "
            "match explanation volume to actual source emphasis."
        )
    return ""


def _core_learning_prose_instructions(a: LectureAnalysis) -> str:
    """
    Core Learning: selective teaching, dependencies, solve-awareness — not a soft summary.
    """
    if a.detected_language == "de":
        org = ""
        if a.is_organizational or a.lecture_kind == "organizational":
            org = (
                "\n\n**Organisatorische Vorlesung:** **knapp und sachlich** — keine Schein-Tiefe, keine Theorie erfinden."
            )
        return (
            "**Core Learning — Auftrag (harte Priorität):**\n"
            "**A) Kernproblem der Vorlesung:** Womit beschäftigt sich die Einheit **wirklich** — welches Problem, "
            "welche Lücke, welche Fragestellung? Warum werden die zentralen Begriffe **jetzt** eingeführt?\n"
            "**B) Abhängigkeiten:** Was muss **zuerst** sitzen? Was baut worauf auf? Was bricht zusammen, wenn ein "
            "Frühschritt fehlt?\n"
            "**C) Gewichtung:** **Meiste Substanz** nur auf **wenige tragende Konzepte** (hohe Topic-Map-Tiefe / "
            "Wiederholung in Vorlesung+Übung). Randthemen: **1–3 Sätze** oder knapper Absatz — **nicht** aufblasen, "
            "nur weil sie vorkommen.\n"
            "**D) Nicht gleichförmig:** Kein Schema „Thema 1 Absatz, Thema 2 Absatz …“ in gleicher Länge und gleicher "
            "Weichheit — **schärfere, längere** Abschnitte nur dort, wo die Vorlesung **wirklich** schwer oder zentral ist.\n"
            "**E) Lösungs-/Prüfungsnähe (wenn Übungs-/Aufgabenanteil):** Welche **Fragetypen**, welche **Unterscheidungen** "
            "beim Lösen, welche **Denkfallen**? Aus der Quelle ableiten — keine erfundenen Aufgaben.\n"
            "**F) Stil:** Zusammenhängende Prosa, klare Übergänge; **Stichpunkte sparsam**. Topic Map **nicht** als "
            "Prosa-Glossar wiederholen — **hier lehren und priorisieren**.\n"
            "**Verbote:** Offensichtliches lang erklären; ausgewogene aber inhaltsleere Absätze; generische Motivation; "
            "Feld-Wikipedia statt **diese Vorlesung**."
            + org
        )
    org_en = ""
    if a.is_organizational or a.lecture_kind == "organizational":
        org_en = "\n\n**Organizational lecture:** **short and factual** — no fake depth."
    return (
        "**Core Learning — mission (strict priority):**\n"
        "**A) Core problem of the lecture:** What is this unit **actually** doing — which question, gap, or job? "
        "Why are the central ideas introduced **here**?\n"
        "**B) Dependencies:** What must come **first**? What builds on what? What fails if an earlier idea is missed?\n"
        "**C) Weighting:** Put **most explanation** into **a few backbone concepts** (high Topic Map depth / repeated in "
        "lecture+exercises). Side topics: **1–3 sentences** or one short paragraph — **do not** pad just because they appear.\n"
        "**D) Not uniform:** Avoid equal-length, equally polished paragraphs for every ###. Go **longer and sharper** only "
        "where the lecture is **genuinely** hard or central.\n"
        "**E) Solve-orientation (when exercises exist):** What **question shapes**, **distinctions**, and **traps** matter? "
        "Ground in the source — do not invent tasks.\n"
        "**F) Style:** Connected prose; **bullets sparingly**. Do **not** re-teach the Topic Map as prose — **teach and "
        "prioritize** here.\n"
        "**Banned:** long explanations of obvious points; balanced but empty prose; generic motivation; field overview "
        "instead of **this lecture**."
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
            "Du bist ein **strenger, selektiver Lern-Tutor** — kein höflicher Zusammenfasser. "
            "Priorisiere was in **dieser** Vorlesung wirklich zählt; verschwende keine Worte auf Offensichtliches oder "
            "generisches Fachbuchgelaber. "
            "Erkläre **Abhängigkeiten**, **typische Fehlvorstellungen** und **wie man denkt** (besonders wo Übungsmaterial existiert). "
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
            "You are a **sharp, selective teaching assistant** — not a polite summarizer. "
            "Prioritize what matters in **this** lecture; do not waste words on obvious points or generic textbook prose. "
            "Explain **dependencies**, **common confusions**, and **how to think** (especially when exercise material exists). "
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
        + _anti_generic_rules(a)
        + "\n\n"
        + _scope_and_topic_rules(a)
        + "\n\n"
        + _profile_rules(a)
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _prompt_quick_overview(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Quick Overview** — **kurze, scharfe Orientierung** (kein Mini-Fassung aller folgenden Abschnitte).\n\n"
            "Beantworte **prägnant** und **vorlesungsspezifisch**:\n"
            "- **Zentrale Frage / Problemstellung:** Worauf zielt die Einheit (nicht „das Thema allgemein“)?\n"
            "- **Was soll die Vorlesung erreichen / etablieren?**\n"
            "- **Warum im Kurs genau hier** — Bezug nur wenn aus Quelle/Kontext sicher?\n"
            "- **Worauf achten** beim Lesen der restlichen Materialien?\n\n"
            "Format:\n"
            "- **4–6** knappe Bullets **oder** sehr kurze Absätze — **Weniger Worte, mehr Kanten**.\n"
            "- **Kein** motivierender Fülltext, **kein** Vorgriff auf Topic Map / Core Learning, **keine** Definitionen/Beispiele.\n"
            "- Jeder Satz muss sich auf **diese** Vorlesung beziehen — sonst streichen.\n\n"
            "Oberste Überschrift exakt: ## Quick Overview"
        )
        extra += (
            _quick_overview_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "quick_overview")
        )
    else:
        extra = (
            "Produce **Quick Overview** — **short, sharp orientation** (not a mini-summary of later sections).\n\n"
            "Answer **tightly** and **lecture-specifically**:\n"
            "- **Central question / problem:** What is this unit actually trying to do?\n"
            "- **What should the lecture establish?**\n"
            "- **Why here in the course** — only if safely grounded in the source/context.\n"
            "- **What to watch for** when reading the rest?\n\n"
            "Format:\n"
            "- **4–6** tight bullets **or** very short paragraphs — **fewer words, more edge**.\n"
            "- No motivational filler, **no** preview of Topic Map / Core Learning, **no** definitions or worked examples.\n"
            "- Every sentence must attach to **this** lecture — otherwise cut.\n\n"
            "Top heading must be exactly: ## Quick Overview"
        )
        extra += (
            _quick_overview_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "quick_overview")
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
            "Erstelle **Topic Map** — **selektive** Strukturkarte (kein Glossar-Dump).\n\n"
            "Ziel: Nur **tragende** Bausteine und **klare** Tiefe — weniger Einträge, dafür schärfere Scores.\n\n"
            "Wichtig:\n"
            "- **Ca. 4–12** Themen — **lieber weniger** als viele gleichwertige Labels. Rand-/Nebenthemen: **weglassen** "
            "oder maximal **ein** kurzer Eintrag mit niedrigem Score.\n"
            "- Wenn Vorlesung **und** Übungen ein Konzept teilen → **höhere** praktische/tragende Gewichtung.\n"
            "- Keine langen Erklärungen — **Karte**, kein zweites Core Learning.\n\n"
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
            + _topic_map_depth_calibration(a)
            + _topic_map_kind_focus(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "topic_map")
        )
    else:
        extra = (
            "Produce **Topic Map** — a **selective** structural map (not a glossary dump).\n\n"
            "Goal: **fewer, sharper** entries with meaningful depth scores — not many equally weighted labels.\n\n"
            "Rules:\n"
            "- **About 4–12** topics — **prefer fewer** over many similar entries. Minor/aside topics: **omit** or at most "
            "**one** short entry with a low score.\n"
            "- If both **lecture and exercises** stress a concept → **raise** practical/backbone weight.\n"
            "- No long explanations — a **map**, not a second Core Learning.\n\n"
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
            + _topic_map_depth_calibration(a)
            + _topic_map_kind_focus(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "topic_map")
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

    map_block += _core_learning_map_depth_calibration(a)

    if a.detected_language == "de":
        extra = (
            "Erstelle **Core Learning** als Haupt-Lernteil — die **eigentliche Erklärung** der Vorlesung "
            "(nicht eine zweite Stichpunktliste).\n\n"
            + _core_learning_prose_instructions(a)
            + "\n\n"
            "Struktur (nur Rahmen):\n"
            "- Oberste Überschrift exakt: ## Core Learning\n"
            "- Hauptthemen mit ### (ein Abschnitt pro Thema); Unterabschnitte #### nur wenn nötig.\n\n"
            "Inhaltliche Tiefe (weiterhin adaptiv):\n"
            "- Hohe Topic-Map-Tiefe: **längere zusammenhängende Erklärung**, stärkere Vernetzung, "
            "typische Missverständnisse und Beispiele **im Erzählfluss** — nicht als Bullet-Wand.\n"
            "- Mittlere Tiefe: klare Absatz-Erklärung + Bedeutung; ggf. ein kurzes Beispiel eingebettet.\n"
            "- Geringe Tiefe: knappe Absätze — genug zum Verstehen, ohne Aufblasen.\n\n"
            "Strikte Regeln:\n"
            "- **Weniger Länge, mehr Nutzen:** Lieber kürzer und schärfer als lang und weich — kein „sicherer“ Überblick.\n"
            "- Nicht jedes ### gleich lang — **Zentrales ausführlich**, Rand kurz.\n"
            "- Topic Map nicht als Glossar wiederholen; **hier lehren** mit Abhängigkeiten und Priorität.\n"
            "- Quick Overview nicht wiederholen.\n"
            "- Nur Vorlesungsinhalt; keine erfundene Erweiterung.\n"
            "- Merklisten → Revision Sheet, nicht hier."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
            + _core_learning_structure_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "core_learning")
        )
    else:
        extra = (
            "Produce **Core Learning** as the main teaching section — the **actual explanation layer** "
            "(not a second bullet outline).\n\n"
            + _core_learning_prose_instructions(a)
            + "\n\n"
            "Structure (skeleton only):\n"
            "- Top heading must be exactly: ## Core Learning\n"
            "- Main topics as ### (one section per topic); #### subheads only if needed.\n\n"
            "Depth (still adaptive):\n"
            "- High Topic Map depth: **longer connected explanation**, stronger linking, pitfalls and examples **in "
            "the narrative flow** — not a wall of bullets.\n"
            "- Medium: clear paragraph explanation + why it matters; maybe one short embedded example.\n"
            "- Low: brief paragraphs — enough to understand, no padding.\n\n"
            "Strict rules:\n"
            "- **Less length, more value:** prefer **shorter and sharper** over long, safe prose.\n"
            "- Not every ### equal length — **deep where central**, brief on side material.\n"
            "- Do not repeat the Topic Map as a glossary; **teach** with dependencies and priority.\n"
            "- Do not repeat Quick Overview.\n"
            "- Lecture-grounded only.\n"
            "- Memorize lists → Revision Sheet, not here."
            + map_block
            + _artifact_technical_addon(a, "core_learning")
            + _core_learning_structure_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "core_learning")
        )
    return sys, extra


def _prompt_revision_sheet(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Revision Sheet** — **kompakte** Merk- und Prüfungsseite (**keine** zweite Erklärphilosophie).\n\n"
            "Aufbau:\n"
            "- **Auswendig lernen**: nur was die Vorlesung **wirklich** einfordert — Formeln, Symbole, Regeln, Fakten.\n"
            "- **Konzeptuell verstehen**: Ideen, die ich **kurz** erklären können muss — **ohne** ausufernde Prosa.\n\n"
            "Regeln:\n"
            "- **Selektiv und kurz** — lieber wenige harte Punkte als lange Listen.\n"
            "- **Keine** ausführlichen Erklärungen (→ Core Learning); nur **Stichworte, Checks, typische Fallen**.\n"
            "- Typische **Frage-/Fehlermuster** und **Unterscheidungen** wenn Übungsmaterial nahelegt.\n"
            "- Keine neuen Themen.\n"
            "- Bullets / Mini-Tabellen; **dichte** Zeilen.\n\n"
            "Oberste Überschrift exakt: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
            + _revision_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "revision_sheet")
        )
    else:
        extra = (
            "Produce a **Revision Sheet** — **compact** cram sheet (**not** a second explainer).\n\n"
            "Structure:\n"
            "- **Memorize**: only what the lecture **actually** demands — rules, formulas, symbols, facts.\n"
            "- **Understand conceptually**: ideas I can explain **briefly** — **no** essay prose.\n\n"
            "Rules:\n"
            "- **Selective and short** — fewer hard hits beat long lists.\n"
            "- **No** long explanations (→ Core Learning); **keywords, checks, typical traps** only.\n"
            "- If exercises suggest it: **question types**, **distinctions**, **common mistakes**.\n"
            "- No new material.\n"
            "- Bullets / compact tables; **tight** lines.\n\n"
            "Top heading must be exactly: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
            + _revision_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "revision_sheet")
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
