"""Interactive topic quizzes (JSON on disk) + lightweight mistake memory (SQLite)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import GENERATION_MODE
from app.db.database import get_connection
from app.services import lecture_service, openai_service
from app.services.generation_readiness import prepare_generation_inputs
from app.services.lecture_analysis import analyze_extracted_text
from app.services.lecture_generation import (
    _layered_material_block,
    _profile_rules,
    _system_prompt,
    _truncate_layered_lecture_exercise,
)
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.slugs import slugify
from app.services.source_manifest import split_combined_extracted_text
from app.services.topic_deep_dive import (
    QUESTION_DIFFICULTIES,
    _truncate_for_prompt,
    deep_dives_root,
    load_topic_map_and_topics,
    read_deep_dive_markdown,
    topic_entry_by_slug,
)

_EXPECTED_N = 10


def interactive_quiz_path(lecture_root: Path, topic_slug: str, difficulty: str) -> Path:
    d = (difficulty or "").strip().lower()
    if d not in QUESTION_DIFFICULTIES:
        d = "medium"
    qdir = deep_dives_root(lecture_root) / "quizzes"
    return qdir / f"{topic_slug}_{d}.json"


def _normalize_concept_key(tag: str) -> str:
    s = slugify((tag or "").strip()) or "concept"
    return s[:120]


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    t = (raw or "").strip()
    if not t:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_quiz_payload(data: dict[str, Any]) -> tuple[bool, str]:
    qs = data.get("questions")
    if not isinstance(qs, list) or len(qs) != _EXPECTED_N:
        return False, f"Expected exactly {_EXPECTED_N} questions."
    seen: set[str] = set()
    for i, q in enumerate(qs):
        if not isinstance(q, dict):
            return False, f"Question {i} is not an object."
        qid = str(q.get("id") or "").strip()
        if not qid or qid in seen:
            return False, f"Question {i} needs a unique id."
        seen.add(qid)
        stem = str(q.get("stem") or "").strip()
        if len(stem) < 8:
            return False, f"Question {i} stem too short."
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            return False, f"Question {i} needs at least 2 options."
        if len(opts) > 8:
            return False, f"Question {i} has too many options."
        for o in opts:
            if not isinstance(o, str) or not str(o).strip():
                return False, f"Question {i} has empty option."
        ci = q.get("correct_index")
        if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
            return False, f"Question {i} correct_index invalid."
        if not str(q.get("concept_tag") or "").strip():
            return False, f"Question {i} needs concept_tag."
        if not str(q.get("why_correct") or "").strip():
            return False, f"Question {i} needs why_correct."
    return True, ""


def get_mistake_rows_for_prompt(lecture_id: int, topic_slug: str, limit: int = 6) -> list[dict[str, Any]]:
    """Highest wrong_count first — fed into quiz generation."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT concept_key, wrong_count, subtopic_slug
            FROM topic_quiz_mistake_stats
            WHERE lecture_id = ? AND topic_slug = ? AND wrong_count > 0
            ORDER BY wrong_count DESC, last_wrong_at DESC
            LIMIT ?
            """,
            (lecture_id, topic_slug, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def list_mistake_summary(lecture_id: int, topic_slug: str, limit: int = 12) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT concept_key, wrong_count, correct_count, last_wrong_at, last_right_at, subtopic_slug
            FROM topic_quiz_mistake_stats
            WHERE lecture_id = ? AND topic_slug = ?
            ORDER BY wrong_count DESC, last_wrong_at DESC
            LIMIT ?
            """,
            (lecture_id, topic_slug, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def _record_answer_stats(
    lecture_id: int,
    topic_slug: str,
    *,
    concept_key: str,
    subtopic_hint: str | None,
    was_correct: bool,
) -> None:
    ck = _normalize_concept_key(concept_key)
    sub = (subtopic_hint or "").strip() or None
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, wrong_count, correct_count FROM topic_quiz_mistake_stats
            WHERE lecture_id = ? AND topic_slug = ? AND concept_key = ?
            """,
            (lecture_id, topic_slug, ck),
        )
        row = cur.fetchone()
        if row is None:
            w, r = (0, 1) if was_correct else (1, 0)
            conn.execute(
                """
                INSERT INTO topic_quiz_mistake_stats
                  (lecture_id, topic_slug, concept_key, subtopic_slug, wrong_count, correct_count, last_wrong_at, last_right_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lecture_id,
                    topic_slug,
                    ck,
                    sub,
                    w,
                    r,
                    None if was_correct else now,
                    now if was_correct else None,
                ),
            )
        else:
            wid = int(row["id"])
            w = int(row["wrong_count"])
            c = int(row["correct_count"])
            if was_correct:
                c += 1
                conn.execute(
                    """
                    UPDATE topic_quiz_mistake_stats
                    SET correct_count = ?, last_right_at = ?, subtopic_slug = COALESCE(?, subtopic_slug)
                    WHERE id = ?
                    """,
                    (c, now, sub, wid),
                )
            else:
                w += 1
                conn.execute(
                    """
                    UPDATE topic_quiz_mistake_stats
                    SET wrong_count = ?, last_wrong_at = ?, subtopic_slug = COALESCE(?, subtopic_slug)
                    WHERE id = ?
                    """,
                    (w, now, sub, wid),
                )
        conn.commit()


def check_answer(
    lecture_id: int,
    topic_slug: str,
    difficulty: str,
    question_id: str,
    selected_index: int,
) -> dict[str, Any]:
    """Validate against saved JSON; update mistake stats; return feedback (no correct answer leaked before)."""
    diff = (difficulty or "").strip().lower()
    if diff not in QUESTION_DIFFICULTIES:
        return {"ok": False, "error": "Invalid difficulty."}

    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return {"ok": False, "error": "Lecture not found."}

    root = lecture_root_from_source_relative(lec["source_file_path"])
    path = interactive_quiz_path(root, topic_slug, diff)
    if not path.is_file():
        return {"ok": False, "error": "No quiz found for this difficulty. Generate one first."}

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "error": "Could not read quiz file."}

    qobj = None
    for q in data.get("questions") or []:
        if isinstance(q, dict) and str(q.get("id")) == str(question_id):
            qobj = q
            break
    if qobj is None:
        return {"ok": False, "error": "Unknown question id."}

    opts = qobj.get("options")
    if not isinstance(opts, list) or not isinstance(qobj.get("correct_index"), int):
        return {"ok": False, "error": "Invalid question data."}

    ci = int(qobj["correct_index"])
    if selected_index < 0 or selected_index >= len(opts):
        return {"ok": False, "error": "Invalid selected option."}

    correct = selected_index == ci
    concept = str(qobj.get("concept_tag") or "topic")
    subh = str(qobj.get("subtopic_hint") or "").strip() or None

    _record_answer_stats(
        lecture_id,
        topic_slug,
        concept_key=concept,
        subtopic_hint=subh,
        was_correct=correct,
    )

    why = str(qobj.get("why_correct") or "").strip()
    review = str(qobj.get("review_if_wrong") or "").strip()
    wrong_note = str(qobj.get("note_on_wrong") or "").strip()

    feedback_parts: list[str] = []
    if correct:
        feedback_parts.append("Correct.")
        if why:
            feedback_parts.append(why)
    else:
        feedback_parts.append("Not quite.")
        if why:
            feedback_parts.append(f"Correct answer: {opts[ci]}. {why}")
        if wrong_note:
            feedback_parts.append(wrong_note)
        if review:
            feedback_parts.append(f"Review: {review}")

    return {
        "ok": True,
        "correct": correct,
        "feedback": "\n".join(feedback_parts).strip(),
    }


def redact_quiz_for_client(data: dict[str, Any]) -> dict[str, Any]:
    """Strip answers for browser JSON."""
    out_q: list[dict[str, Any]] = []
    for q in data.get("questions") or []:
        if not isinstance(q, dict):
            continue
        out_q.append(
            {
                "id": str(q.get("id")),
                "type": str(q.get("type") or "mcq"),
                "stem": str(q.get("stem") or ""),
                "options": list(q.get("options") or []),
                "concept_tag": str(q.get("concept_tag") or ""),
            }
        )
    return {
        "version": data.get("version", 1),
        "difficulty": data.get("difficulty"),
        "generated_at": data.get("generated_at"),
        "topic_title": data.get("topic_title"),
        "questions": out_q,
    }


def load_interactive_quiz(lecture_root: Path, topic_slug: str, difficulty: str) -> dict[str, Any] | None:
    p = interactive_quiz_path(lecture_root, topic_slug, difficulty)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def run_generate_interactive_quiz(lecture_id: int, topic_slug: str, difficulty: str) -> tuple[bool, str]:
    if not openai_service.is_openai_configured():
        return False, "OpenAI is not configured. Set OPENAI_API_KEY in your .env file."

    diff = (difficulty or "").strip().lower()
    if diff not in QUESTION_DIFFICULTIES:
        return False, "Invalid difficulty."

    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    prep = prepare_generation_inputs(lecture_id)
    if not prep.ok or not prep.payload:
        return False, prep.reason

    lecture_root = lecture_root_from_source_relative(lec["source_file_path"])
    _tm, topics, err = load_topic_map_and_topics(lecture_root)
    if err:
        return False, err
    entry = topic_entry_by_slug(topics, topic_slug)
    if not entry:
        return False, "Unknown topic slug."

    parent_md = read_deep_dive_markdown(lecture_root, topic_slug)
    if not (parent_md or "").strip():
        return False, "Generate the topic deep dive first."

    topic_title = entry["title"]
    full_text = prep.payload["extracted_text"]
    lecture_core, exercise_raw, _ = split_combined_extracted_text(full_text)
    if not (lecture_core or "").strip():
        lecture_core = full_text
    analysis = analyze_extracted_text(
        full_text,
        generation_mode=GENERATION_MODE,
        lecture_core_text=lecture_core,
        exercise_text=exercise_raw,
    )
    lc, ex = _truncate_layered_lecture_exercise(lecture_core, exercise_raw)
    material_block = _layered_material_block(
        prep.payload["course_name"],
        prep.payload["lecture_title"],
        lc,
        ex,
        language_is_de=analysis.detected_language == "de",
        is_organizational=analysis.is_organizational,
    )
    material_block = _truncate_for_prompt(material_block, 72_000)

    mistakes = get_mistake_rows_for_prompt(lecture_id, topic_slug, 8)
    mistake_lines = ""
    if mistakes:
        lines = [
            f"- {m['concept_key']} (missed {m['wrong_count']}x)"
            + (f" — subtopic: {m['subtopic_slug']}" if m.get("subtopic_slug") else "")
            for m in mistakes
        ]
        mistake_lines = (
            "\n### Previous struggles (include 1–2 questions that probe these if still relevant)\n"
            + "\n".join(lines)
        )

    if analysis.detected_language == "de":
        diff_line = {
            "easy": "leicht: Grundlagen, Definitionen, direkte Anwendung.",
            "medium": "mittel: typische Prüfungsfragen, Unterscheidungen auf Kursniveau.",
            "hard": "schwer: anspruchsvoll im Rahmen dieser Veranstaltung — keine externe Theorie.",
        }.get(diff, "mittel")
        schema_hint = (
            "Antwort **nur als ein JSON-Objekt** (kein Markdown außerhalb). Schema:\n"
            '{"questions":[{"id":"q0","type":"mcq","stem":"...","options":["A","B","C","D"],'
            '"correct_index":0,"concept_tag":"Kurzlabel","subtopic_hint":"optional",'
            '"why_correct":"kurz","review_if_wrong":"was wiederholen","note_on_wrong":"optional"}]}\n'
            f"Genau {_EXPECTED_N} Fragen. Nur Multiple-Choice (4 Optionen) oder Wahr/Falsch (2 Optionen)."
        )
        user_extra = (
            f"Kurs: {prep.payload['course_name']}\nVorlesung: {prep.payload['lecture_title']}\n"
            f"Thema: {topic_title}\nSchwierigkeit: {diff_line}\n{mistake_lines}\n\n"
            f"{schema_hint}\n\n### Topic deep dive\n\n{_truncate_for_prompt(parent_md, 9000)}\n\n"
            f"### Quelle (Vorlesung + Übung)\n\n{material_block}\n"
        )
    else:
        diff_line = {
            "easy": "easy: core concepts and direct recall.",
            "medium": "medium: typical exam-style reasoning for this course.",
            "hard": "hard: challenging within this course — not trivia from outside.",
        }.get(diff, "medium")
        schema_hint = (
            "Return **only one JSON object** (no markdown outside). Schema:\n"
            '{"questions":[{"id":"q0","type":"mcq","stem":"...","options":["A","B","C","D"],'
            '"correct_index":0,"concept_tag":"short label","subtopic_hint":"optional",'
            '"why_correct":"brief","review_if_wrong":"what to revisit","note_on_wrong":"optional"}]}\n'
            f"Exactly {_EXPECTED_N} questions. Multiple choice (4 options) or true/false (2 options) only."
        )
        user_extra = (
            f"Course: {prep.payload['course_name']}\nLecture: {prep.payload['lecture_title']}\n"
            f"Topic: {topic_title}\nDifficulty: {diff_line}\n{mistake_lines}\n\n"
            f"{schema_hint}\n\n### Topic deep dive\n\n{_truncate_for_prompt(parent_md, 9000)}\n\n"
            f"### Sources (lecture + exercises)\n\n{material_block}\n"
        )

    system = (
        _system_prompt(analysis)
        + "\n\nYou produce an **interactive quiz** as strict JSON. Ground every question in the deep dive and sources. "
          "concept_tag must name the tested idea (for mistake tracking). "
          "Do not include prose outside JSON.\n"
        + _profile_rules(analysis)
    )

    ok, raw_out, err_msg = openai_service.chat_completion_markdown(
        system_prompt=system,
        user_prompt=user_extra,
        max_tokens=6500,
    )
    if not ok:
        return False, err_msg or "Generation failed."

    parsed = _extract_json_object(raw_out)
    if not parsed:
        return False, "Could not parse quiz JSON from the model."

    parsed["difficulty"] = diff
    parsed["topic_title"] = topic_title
    parsed["version"] = 1
    parsed["generated_at"] = datetime.now(timezone.utc).isoformat()

    valid, verr = _validate_quiz_payload(parsed)
    if not valid:
        return False, f"Quiz validation failed: {verr}"

    qdir = interactive_quiz_path(lecture_root, topic_slug, diff).parent
    qdir.mkdir(parents=True, exist_ok=True)
    out_path = interactive_quiz_path(lecture_root, topic_slug, diff)
    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, f"Interactive quiz ({diff}) saved — {_EXPECTED_N} questions."
