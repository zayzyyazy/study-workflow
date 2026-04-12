"""Generate study materials (Markdown) into outputs/ and artifacts table."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from app.services import artifact_service, lecture_service, openai_service
from app.services.generation_readiness import prepare_generation_inputs
from app.services import lecture_meta
from app.services.lecture_analysis import LectureAnalysis, analyze_extracted_text
from app.services.lecture_paths import lecture_root_from_source_relative

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
) -> tuple[bool, str, str]:
    user = extra_user_instruction.strip() + "\n\n" + _base_user_block(course_name, lecture_title, lecture_text)
    return openai_service.chat_completion_markdown(system_prompt=system, user_prompt=user)


def _profile_rules(a: LectureAnalysis) -> str:
    """Extra system rules for math/code/mixed; language-aware."""
    if a.detected_language == "de":
        parts: list[str] = []
        if a.content_profile in ("math", "mixed") or a.has_formulas:
            parts.append(
                "Mathematischer Inhalt: Formeln und Symbole aus der Quelle möglichst exakt übernehmen; "
                "nicht in vage Prosa umformulieren, wenn eine klare Darstellung möglich ist. "
                "Keine neuen Gleichungen erfinden. Bei Erklärungen die Bedeutung der Variablen nennen, wenn möglich. "
                "Markdown mit `$...$` oder ähnlich nutzen, wenn die Quelle mathematische Notation enthält."
            )
        if a.content_profile in ("code", "mixed") or a.has_code:
            parts.append(
                "Code: relevante Ausschnitte in Markdown-Fenced-Codeblöcken (dreifache Backticks) wiedergeben. "
                "Code nicht willkürlich umschreiben oder \"reparieren\"; Funktionsnamen, Variablen und APIs wie in der Quelle belassen. "
                "In verständlicher Sprache erklären, was der Code bewirkt."
            )
        if a.content_profile == "mixed":
            parts.append(
                "Gemischter Inhalt (Mathe und Code): beides klar trennen; Formeln und Code jeweils angemessen formatieren."
            )
        if not parts:
            parts.append(
                "Fachbegriffe aus der Quelle beibehalten, wenn sie dort üblich sind; nicht unnötig vereinfachen."
            )
        return "\n".join(parts)
    parts_en: list[str] = []
    if a.content_profile in ("math", "mixed") or a.has_formulas:
        parts_en.append(
            "Math content: preserve formulas and symbols from the source as exactly as possible; "
            "do not rewrite them into vague prose when a clear formula can stay. "
            "Do not invent equations not present in the source. When explaining, name what each variable means when possible. "
            "Use readable Markdown math-style notation (e.g. `$...$`) when it matches the source."
        )
    if a.content_profile in ("code", "mixed") or a.has_code:
        parts_en.append(
            "Code: keep useful snippets in fenced Markdown code blocks (triple backticks). "
            "Do not arbitrarily rewrite or \"fix\" code unless framing it as commentary; keep identifiers and APIs as in the source. "
            "Explain what the code does in plain language."
        )
    if a.content_profile == "mixed":
        parts_en.append(
            "Mixed content: treat both formulas and code carefully and keep them visually distinct."
        )
    if not parts_en:
        parts_en.append(
            "Keep standard technical terms from the source accurate; avoid oversimplifying vocabulary."
        )
    return "\n".join(parts_en)


def _system_prompt(a: LectureAnalysis) -> str:
    if a.detected_language == "de":
        base = (
            "Du schreibst klare Lernnotizen in Markdown. "
            "Verwende Überschriften (##, ###), Aufzählungen wo sinnvoll und **Fettdruck** für zentrale Begriffe. "
            "Sei nur an der Vorlesung orientiert; erfinde keine Fakten. "
            "Die folgende Sprach-/Inhaltsanalyse dient der Formatierung; der Vorlesungstext selbst ist maßgeblich."
        )
        analysis = (
            f"Voranalyse: Ausgaben durchgehend auf **Deutsch**. "
            f"Inhaltstyp: {a.content_profile}. "
            f"Quelle enthält erkennbare Formeln: {'ja' if a.has_formulas else 'nein'}. "
            f"Quelle enthält erkennbaren Code: {'ja' if a.has_code else 'nein'}."
        )
    else:
        base = (
            "You write clear study notes in Markdown. "
            "Use headings (##, ###), bullet lists where helpful, and bold for key terms. "
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


def _prompt_glossary(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle einen Abschnitt **Glossar**.\n"
            "- Liste wichtige Begriffe aus dieser Vorlesung mit kurzen Definitionen.\n"
            "- Halte es knapp und gut scannbar (Tabelle oder Aufzählung).\n"
            "- Beginne mit genau einer Überschrift ## Glossar."
        )
    else:
        extra = (
            "Produce a **Glossary** section.\n"
            "- List important terms from this lecture with short definitions.\n"
            "- Keep it concise and easy to scan (table or bullet list).\n"
            "- Start with a single ## Glossary heading."
        )
    return sys, extra


def _prompt_summary(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle eine **Zusammenfassung** dieser Vorlesung.\n"
            "- Verständliche Sprache; Hauptthemen und wichtige Erkenntnisse.\n"
            "- Verwende ## Zusammenfassung als oberste Überschrift, dann kurze Unterabschnitte bei Bedarf."
        )
    else:
        extra = (
            "Produce a **Summary** of this lecture.\n"
            "- Plain language; main themes and important takeaways.\n"
            "- Use ## Summary as the top heading, then short subsections if needed."
        )
    return sys, extra


def _prompt_topics(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Themen mit Kurzerklärungen**.\n"
            "- Teile die Vorlesung in wenige Hauptthemen.\n"
            "- Zu jedem Thema eine kurze Erklärung zum schnellen Wiederholen.\n"
            "- Beginne mit ## Themen und Kurzerklärungen."
        )
    else:
        extra = (
            "Produce **Topic explanations**.\n"
            "- Break the lecture into a few main topics.\n"
            "- For each topic, give a short explanation good for quick review.\n"
            "- Start with ## Topic explanations."
        )
    return sys, extra


def _prompt_deep_dive(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle eine **Vertiefung**.\n"
            "- Wähle die wichtigsten Konzepte oder Themen aus dieser Vorlesung.\n"
            "- Erkläre sie ausführlicher; kurze Beispiele nur wenn hilfreich.\n"
            "- Beginne mit ## Vertiefung."
        )
    else:
        extra = (
            "Produce a **Deep dive**.\n"
            "- Pick the most important concepts or topics from this lecture.\n"
            "- Explain them in more depth; add brief examples when helpful.\n"
            "- Start with ## Deep dive."
        )
    return sys, extra


def _prompt_connections(a: LectureAnalysis) -> tuple[str, str]:
    sys = _system_prompt(a)
    if a.detected_language == "de":
        extra = (
            "Erstelle **Zusammenhänge** nur innerhalb dieser Vorlesung.\n"
            "- Zeige, wie Ideen zusammenhängen: Ursache/Wirkung, Vergleiche, Abhängigkeiten, Ablauf.\n"
            "- Beziehe dich nicht auf andere Vorlesungen oder Kurse.\n"
            "- Beginne mit ## Zusammenhänge."
        )
    else:
        extra = (
            "Produce **Connections** within this lecture only.\n"
            "- Show how ideas relate: causes/effects, comparisons, dependencies, sequence.\n"
            "- Do NOT reference other lectures or courses.\n"
            "- Start with ## Connections."
        )
    return sys, extra


GENERATION_STEPS: list[tuple[str, str, Callable[[LectureAnalysis], tuple[str, str]]]] = [
    ("glossary", "01_glossary.md", _prompt_glossary),
    ("summary", "02_summary.md", _prompt_summary),
    ("topic_explanations", "03_topic_explanations.md", _prompt_topics),
    ("deep_dive", "04_deep_dive.md", _prompt_deep_dive),
    ("connections", "05_connections.md", _prompt_connections),
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
    for artifact_type, filename, prompt_fn in GENERATION_STEPS:
        system, user_extra = prompt_fn(analysis)
        ok, md, err = _run_one(
            system=system,
            extra_user_instruction=user_extra,
            course_name=course_name,
            lecture_title=lecture_title,
            lecture_text=lecture_text,
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
        out_path.write_text(md + "\n", encoding="utf-8")
        written_paths.append(out_path)
        rel = lecture_meta.relative_to_app(out_path)
        results.append((artifact_type, rel))

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
