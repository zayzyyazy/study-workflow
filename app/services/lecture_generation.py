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
# Longer notes are truncated with a clear notice (see README).
MAX_LECTURE_CHARS = 120_000


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
            "- Keep function names, variables, APIs, and keywords exactly as in the source; do not silently rewrite or “fix” them.\n"
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
        if step == "glossary" and wm:
            bullets.append(
                "- Glossar: Begriffe mit mathematischen Symbolen in `$...$` setzen; bei Bedarf `$$...$$` für längere Ausdrücke. "
                "Keine losen Unterstriche für Indizes."
            )
        if step == "glossary" and wc:
            bullets.append(
                "- Glossar: Code-Identifier und Schlüsselwörter in `Backticks`; mehrzeilige Signaturen in fenced Blocks."
            )
        if step == "teach_me" and wm:
            bullets.append(
                "- Lernen/Erklären: Formeln in `$...$` / `$$...$$`; pro Hauptthema klar mit ##/### gliedern."
            )
        if step == "teach_me" and wc:
            bullets.append("- Lernen/Erklären: Code in fenced Blocks zeigen, wenn die Vorlesung Code nutzt.")
        if step == "examples_and_solutions" and wm:
            bullets.append(
                "- Ausgearbeitete Beispiele: jeder Rechenschritt mit `$...$`; kurz begründen, warum der Schritt folgt."
            )
        if step == "examples_and_solutions" and wc:
            bullets.append(
                "- Ausgearbeitete Beispiele: Code schrittweise oder blockweise erklären; Einrückung beibehalten."
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
    if step == "glossary" and wm:
        bullets_en.append(
            "- Glossary: put mathematical symbols/expressions in `$...$`; use `$$...$$` for larger expressions when needed. "
            "Avoid raw underscores for subscripts outside math mode."
        )
    if step == "glossary" and wc:
        bullets_en.append(
            "- Glossary: put code identifiers/keywords in `backticks`; use fenced blocks for multi-line signatures."
        )
    if step == "teach_me" and wm:
        bullets_en.append(
            "- Teach Me: use `$...$` / `$$...$$` for formulas; structure each main topic with ##/### headings."
        )
    if step == "teach_me" and wc:
        bullets_en.append("- Teach Me: use fenced code blocks when the lecture uses code.")
    if step == "examples_and_solutions" and wm:
        bullets_en.append(
            "- Worked examples: show each step; justify steps briefly; use `$...$` for all math."
        )
    if step == "examples_and_solutions" and wc:
        bullets_en.append(
            "- Worked examples: walk through code block by block or line by line where helpful; preserve indentation."
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


# --- Prompts: structured, predictable Markdown (language + profile via _system_prompt) ---


def _prompt_quick_overview(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Quick Overview** als kurze Orientierung vor allen Details.\n"
            "- 5–9 Bullet-Points oder kurze Absätze, klar und ohne Füllsätze.\n"
            "- Beantworte: Worum geht es? Was ist die Hauptidee? Warum ist das wichtig?\n"
            "- Beziehe den Überblick auf den tatsächlichen Vorlesungsinhalt; keine generischen Aussagen.\n"
            "- Wenn in der Vorlesung erkennbar: kurz die Verbindung zu angrenzenden Themen nennen.\n"
            "- Keine langen Definitionen und keine ausgearbeiteten Beispiele hier.\n\n"
            "Oberste Überschrift exakt: ## Quick Overview"
        )
    else:
        extra = (
            "Produce **Quick Overview** as the short orientation layer before details.\n"
            "- 5–9 concise bullets or short paragraphs; no generic filler.\n"
            "- Answer: what is this lecture about, what is the big idea, and why does it matter.\n"
            "- Ground statements in the actual lecture text, not generic textbook claims.\n"
            "- If inferable from the lecture, mention nearby/related topics briefly.\n"
            "- Do not expand into full definitions or worked examples here.\n\n"
            "Top heading must be exactly: ## Quick Overview"
        )
    return sys, extra


def _prompt_glossary(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle ein **Glossar** (nur Referenz, knapp).\n"
            "- Nur zentrale Begriffe dieser Vorlesung mit sehr kurzen Definitionen.\n"
            "- Tabelle oder Aufzählung; keine langen Erklärungen — die kommen in „Lernen“.\n"
            "- Beginne mit genau einer Überschrift ## Glossar."
            + _artifact_technical_addon(a, "glossary")
        )
    else:
        extra = (
            "Produce a **Glossary** (reference only — keep it tight).\n"
            "- Only key terms from this lecture with very short definitions.\n"
            "- Table or bullet list; do not write long explanations here (those belong in Teach Me).\n"
            "- Start with a single ## Glossary heading."
            + _artifact_technical_addon(a, "glossary")
        )
    return sys, extra


def _prompt_teach_me(a: LectureAnalysis) -> tuple[str, str]:
    """Main teaching file — tutor-style, per-topic structure."""
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Teach Me** als Haupt-Lernteil.\n"
            "Schreibe wie ein Tutor und richte die Tiefe pro Thema an der Vorlesungs-Tiefe aus.\n\n"
            "Struktur:\n"
            "- Oberste Überschrift exakt: ## Teach Me\n"
            "- Danach Hauptthemen mit ##-Überschriften.\n"
            "- Unterthemen bei Bedarf mit ###.\n\n"
            "Adaptive Tiefe (wichtig):\n"
            "- Wenn Thema nur kurz erwähnt: 3-5 knappe Bullet-Points reichen.\n"
            "- Wenn Thema zentral/wiederholt/formalisiert: deutlich tiefer erklären (Intuition + formal + Anwendung + typische Verwechslung).\n"
            "- Wenn Thema strukturell zentral mit Definitionen/Notation/Beispielen: besonders gründlich und mit klaren Zusammenhängen zu anderen Themen.\n"
            "- Nicht jedes Thema gleich lang behandeln.\n\n"
            "Inhalt pro Thema (wo passend):\n"
            "- Intuition in klarer Sprache.\n"
            "- Formale Bedeutung/Regel/Notation aus der Vorlesung.\n"
            "- Warum es wichtig ist.\n"
            "- Wie man es in Aufgaben erkennt und anwendet.\n"
            "- Typische Falle nur wenn plausibel.\n\n"
            "Redundanzregeln:\n"
            "- Keine Glossar-Definitionen in voller Länge wiederholen.\n"
            "- Nur Inhalte aus der Vorlesung nutzen; keine künstliche Curriculum-Erweiterung."
            + _artifact_technical_addon(a, "teach_me")
        )
    else:
        extra = (
            "Produce **Teach Me** as the main learning file.\n"
            "Write like a tutor and adapt depth to topic importance in this lecture.\n\n"
            "Structure:\n"
            "- Top heading must be exactly: ## Teach Me\n"
            "- Then main topics with ## headings.\n"
            "- Use ### only when it helps readability.\n\n"
            "Adaptive depth (critical):\n"
            "- Briefly mentioned/supporting topics: keep short and concise.\n"
            "- Core/repeated/formalized topics: explain in more depth (intuition + formal meaning + usage cues + pitfalls).\n"
            "- Deep structural topics with heavy lecture emphasis: provide the deepest treatment and relationships to nearby concepts.\n"
            "- Do not force equal depth for every topic.\n\n"
            "Per-topic content when relevant:\n"
            "- Intuition in plain language.\n"
            "- Formal definition/rule/notation from the lecture.\n"
            "- Why it matters.\n"
            "- How to recognize and apply it.\n"
            "- Common confusion only if plausible.\n\n"
            "Anti-repetition rules:\n"
            "- Do not restate glossary definitions in full.\n"
            "- Stay grounded in lecture scope; avoid inventing a broader curriculum."
            + _artifact_technical_addon(a, "teach_me")
        )
    return sys, extra


def _prompt_examples_and_solutions(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Examples and Solutions** als Anwendungs-Teil.\n"
            "- Fokus auf Anwenden und Lösen, nicht auf Definitionen wiederholen.\n"
            "- Nutze bevorzugt Muster/Beispiele aus der Vorlesung, wenn erkennbar.\n"
            "- Für technische/mathematische Inhalte: schrittweise Lösung mit knapper Begründung je Schritt.\n"
            "- Für konzeptuelle Inhalte: konkrete Fallanwendungen/Interpretationen mit klarer Argumentation.\n"
            "- Umfang an Vorlesungstiefe anpassen: Kern-Themen dürfen mehrere Beispiele haben, Nebenpunkte nur kurz.\n\n"
            "Oberste Überschrift exakt: ## Examples and Solutions"
            + _artifact_technical_addon(a, "examples_and_solutions")
        )
    else:
        extra = (
            "Produce **Examples and Solutions** as the application layer.\n"
            "- Focus on doing/applying, not repeating definitions.\n"
            "- Prefer example patterns that are clearly present in the lecture.\n"
            "- For technical/math/code content: step-by-step solutions with brief reasoning per step.\n"
            "- For conceptual lectures: concrete applied cases with clear interpretation steps.\n"
            "- Adapt depth to lecture emphasis: major topics may have multiple examples, side topics stay concise.\n\n"
            "Top heading must be exactly: ## Examples and Solutions"
            + _artifact_technical_addon(a, "examples_and_solutions")
        )
    return sys, extra


def _prompt_revision_sheet(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Revision Sheet** als kurze Wiederholungsseite.\n"
            "- Kurz, skimmbar, prüfungsnah.\n"
            "- Trenne klar: **Memorize** vs **Understand**.\n"
            "- Fokus auf Kernregeln/Symbole/Formeln/Fakten aus der Vorlesung.\n"
            "- Keine langen Erklärungen und keine neuen Themen.\n\n"
            "Oberste Überschrift exakt: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
        )
    else:
        extra = (
            "Produce a **Revision Sheet** — short, exam-friendly, skimmable.\n"
            "- Key rules, formulas, facts (use `$...$` / `$$...$$` for math).\n"
            "- Clearly separate **memorize** vs **understand conceptually**.\n"
            "- Bullet lists; avoid long paragraphs.\n"
            "- No new material; only what the lecture supports.\n\n"
            "Top heading must be exactly: ## Revision Sheet"
            + _artifact_technical_addon(a, "revision_sheet")
        )
    return sys, extra


GENERATION_STEPS: list[
    tuple[str, str, Callable[[LectureAnalysis], tuple[str, str]], int]
] = [
    ("quick_overview", "01_quick_overview.md", _prompt_quick_overview, 3072),
    ("glossary", "02_glossary.md", _prompt_glossary, 4096),
    ("teach_me", "03_teach_me.md", _prompt_teach_me, 8192),
    ("examples_and_solutions", "04_examples_and_solutions.md", _prompt_examples_and_solutions, 8192),
    ("revision_sheet", "05_revision_sheet.md", _prompt_revision_sheet, 6144),
]


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
    for artifact_type, filename, prompt_fn, max_tok in GENERATION_STEPS:
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

    pack_body = cleanup_generated_markdown(build_study_pack_markdown(outputs_dir))
    pack_path = outputs_dir / "06_study_pack.md"
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
