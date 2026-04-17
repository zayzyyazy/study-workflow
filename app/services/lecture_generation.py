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
            "Anti-Fülltext (verbindlich):\n"
            "- Nicht jedes Thema gleich tief behandeln — **Tiefe proportional** zur Rolle in der Vorlesung.\n"
            "- Keine künstliche Vollständigkeit: lieber **präzise und kurz** wo die Quelle dünn ist.\n"
            "- Keine wiederholten Glossar-Definitionen in jedem Abschnitt; **einmal klar, dann verbinden**.\n"
            "- Keine erfundenen „Alltagsbeispiele“, nur wenn sie wirklich helfen.\n"
            "- Keine gleichförmigen Abschnitte — **variiere Struktur und Länge** nach inhaltlicher Bedeutung.\n"
            "- Offensichtliches nicht überdehnen; **Kernideen** dort klar machen, wo die Vorlesung Gewicht legt."
        )
    return (
        "Anti-generic behavior (mandatory):\n"
        "- Do **not** explain every topic at the same depth — scale depth to **importance in the lecture**.\n"
        "- Avoid fake completeness: prefer **short and precise** when the source is thin.\n"
        "- Do not repeat glossary-style definitions in every section; **define once, then reuse**.\n"
        "- Do not invent shallow “real-life” examples unless they genuinely clarify.\n"
        "- Avoid uniform sections — **vary structure and length** by conceptual weight.\n"
        "- Do not over-expand trivial points; spend words on what the lecture **actually emphasizes**."
    )


def _scope_and_topic_rules(a: LectureAnalysis) -> str:
    """Outputs must feel specific to this course/lecture, not interchangeable summaries."""
    if a.detected_language == "de":
        return (
            "Bezug & Originalität (verbindlich):\n"
            "- Formulierungen und Schwerpunkte müssen **erkennbar zu genau dieser Vorlesung** passen — "
            "nicht wie ein generisches Fachbuch.\n"
            "- **Kurs- und Themenabhängigkeit:** Begriffe, Beispiele und Gewichtung aus der **Quelle** ableiten; "
            "keine austauschbare „Standard-Erklärung“.\n"
            "- **Rhythmus und Stil** an die Vorlesung anpassen — nicht jede Einheit gleich klingen lassen."
        )
    return (
        "Scope & originality (mandatory):\n"
        "- Wording and emphasis must **clearly fit this specific lecture** — not a generic textbook.\n"
        "- **Course/topic dependence:** derive terms, examples, and weighting from the **source**; "
        "avoid interchangeable default explanations.\n"
        "- **Vary rhythm and tone** to match the lecture — do not make every unit sound the same."
    )


def _exercise_application_addon(a: LectureAnalysis, step: str) -> str:
    """Stronger solving/practice behavior when exercise-like sources or task language are present."""
    if not (a.has_exercise_material or a.problem_solving_emphasis or a.practical_density == "high"):
        return ""
    if a.detected_language == "de":
        base = (
            "\n\n**Übungs-/Anwendungsbezug (Heuristik: zusätzliche Quellen oder Aufgabenanteil erkannt):**\n"
            "- Nutze **Aufgaben/Übungsblätter** in der Quelle, um zu erkennen, **welche Konzepte praktisch prüfungsrelevant** "
            "sind und **wie Fragen typischerweise gestellt** werden.\n"
            "- Themen, die in **Vorlesung und Übung** vorkommen, als **tragender** behandeln — nicht nur nennen.\n"
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
        "- Use **problem sets / worksheets** in the source to infer **what is practically important** and "
        "**how questions are typically phrased**.\n"
        "- Topics appearing in **both lecture and exercises** deserve **more weight** — not just a name-check.\n"
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
    Narrative-first rules for Core Learning: connected explanation, not bullet dumps.
    Adaptive depth preserved — deeper topics get richer prose and transitions, not more list items.
    """
    if a.detected_language == "de":
        org = ""
        if a.is_organizational or a.lecture_kind == "organizational":
            org = (
                "\n\n**Organisatorische Vorlesung:** sachlich und **kurz in Absätzen**; kein künstlich ausgedehntes "
                "„Tutor“-Pathos — Fokus auf Abläufe, Regeln, was Studierende tun müssen."
            )
        return (
            "**Schreibweise (verbindlich):**\n"
            "- Schreibe wie ein **Tutor**, der die Vorlesung **durchgehend erklärt** — **nicht** wie ein Werkzeug, "
            "das Folien in Stichpunkte zerlegt.\n"
            "- **Standard sind kurze Absätze** mit klarem Faden: eine Idee führt zur nächsten; nutze **Übergänge** "
            "(„Daraus folgt …“, „Das setzt voraus …“, „Damit wird verständlich, warum …“).\n"
            "- **Stichpunkte sparsam:** nur wenn sie wirklich helfen — z. B. Prinzipienlisten, Fälle/Typen, "
            "kurze Vergleiche, oder echte **Schritt-für-Schritt**-Verfahren. **Nicht** jedes Thema als Bullet-Kette.\n"
            "- **Lecture-Flow:** Pro Hauptthema klar machen: **was** das Thema ist, **wie** es zum vorherigen Thema "
            "passt, **warum** es an dieser Stelle der Vorlesung steht, **worauf** spätere Teile aufbauen.\n"
            "- **Topic Map vs. Core Learning:** Die Topic Map ist die **Karte** — hier **nicht** die gleichen "
            "Einträge als flache Bullet-Liste mit Mini-Definitionen wiederholen. Stattdessen **durch die Themen "
            "hindurch erklären**, sodass man den **Aufbau der Vorlesung** spürt.\n"
            "- **Adaptive Tiefe:** Tiefere Scores bedeuten **mehr erklärende Substanz, bessere Vernetzung und "
            "Beispiele im Fluss** — nicht „mehr Bullets“.\n"
            "- Vermeide **glossarartige** Wiederholung: Begriffe **einmal sauber einführen**, danach **verbinden**."
            + org
        )
    org_en = ""
    if a.is_organizational or a.lecture_kind == "organizational":
        org_en = (
            "\n\n**Organizational lecture:** keep it **practical and compact in paragraphs**; no fake extended "
            "“tutor theater” — focus on flows, rules, and what students must do."
        )
    return (
        "**Writing style (mandatory):**\n"
        "- Write like a **tutor explaining the lecture end-to-end** — **not** like a slide extractor turning "
        "everything into bullets.\n"
        "- **Default to short paragraphs** with a clear thread: one idea leads to the next; use **transitions** "
        "(“This means…”, “That relies on…”, “Once you see why…”).\n"
        "- **Use bullets sparingly** — only when they truly help: principles, types/categories, short comparisons, "
        "or genuine **step-by-step** procedures. **Do not** default every topic to a bullet list.\n"
        "- **Lecture flow:** For each major topic, make explicit: **what** it is, **how** it connects to what came "
        "before, **why** it appears at this point, and **what** later parts depend on.\n"
        "- **Topic Map vs Core Learning:** The Topic Map is the **map** — do **not** repeat it as a flat bullet "
        "glossary with mini-definitions. **Teach through** the topics so the **structure of the lecture** is felt.\n"
        "- **Adaptive depth:** Higher depth means **richer explanation, stronger links, examples in-flow** — "
        "**not** “more bullets”.\n"
        "- Avoid **glossary repetition**: introduce terms **once clearly**, then **reuse and connect**."
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
            "Du bist ein Lern-Coach / Tutor: Ziel ist echtes Üben und Verstehen — nicht nur eine kurze Zusammenfassung. "
            "Erkläre verständlich, strukturiere für Wiederholung und Prüfung, nenne typische Fehler und Checks. "
            "Schreibe in Markdown mit Überschriften (##, ###), Listen und **Fettdruck** für zentrale Begriffe. "
            "Nur an der Vorlesung orientiert; nichts erfinden. "
            "Die Sprach-/Inhaltsanalyse unten steuert die Formatierung; der Vorlesungstext ist maßgeblich."
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
            "You are a learning coach / tutor: prioritize real teaching — not a thin abstract summary. "
            "Explain clearly, structure for review and exams, surface common mistakes and self-checks. "
            "Use Markdown with headings (##, ###), lists, and bold for key terms. "
            "Be accurate to the lecture only; do not invent facts. "
            "The lecture analysis below guides formatting; the lecture text itself is authoritative."
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
        extra += (
            _quick_overview_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "quick_overview")
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
            + _topic_map_depth_calibration(a)
            + _topic_map_kind_focus(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "topic_map")
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
            "- Nicht jedes Thema gleich lang — Tiefe folgt der Vorlesung und den Scores.\n"
            "- Topic Map nicht als Glossar wiederholen; **hier lehren**, nicht erneut nur benennen.\n"
            "- Keine Quick-Overview wiederholen.\n"
            "- Nur Vorlesungsinhalt; keine erfundene Erweiterung.\n"
            "- Reine Auswendig-Merklisten → Revision Sheet, nicht hier."
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
            "- Do not give every topic equal length — follow the lecture and scores.\n"
            "- Do not repeat the Topic Map as a glossary; **teach** here, not re-label.\n"
            "- Do not repeat Quick Overview.\n"
            "- Stay grounded in the lecture only.\n"
            "- Pure memorize lists belong in the Revision Sheet, not here."
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
            + _revision_kind_addon(a)
            + _example_policy_line(a)
            + _exercise_application_addon(a, "revision_sheet")
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
