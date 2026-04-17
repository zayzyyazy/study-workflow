"""On-demand Topic Deep Dive pages — Markdown files under outputs/topic_deep_dives/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import GENERATION_MODE
from app.services import lecture_service, openai_service
from app.services.generation_markdown_cleanup import cleanup_generated_markdown
from app.services.generation_readiness import prepare_generation_inputs
from app.services.lecture_analysis import LectureAnalysis, analyze_extracted_text
from app.services.lecture_generation import (
    _analysis_signal_lines,
    _artifact_technical_addon,
    _exercise_application_addon,
    _system_prompt,
    _truncate_layered_lecture_exercise,
    _example_policy_line,
    _layered_material_block,
)
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.slugs import slugify
from app.services.source_manifest import split_combined_extracted_text
from app.services.study_output_paths import resolve_existing_output

TOPIC_DEEP_DIVES_DIR = "topic_deep_dives"
INDEX_FILENAME = "index.json"
RECOMMENDED_PRIORITY_MIN = 7

_H3_TOPIC = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_PRIORITY = re.compile(
    r"\*\*(?:Priorität|Priority):\*\*\s*(\d{1,2})\s*/\s*10",
    re.IGNORECASE,
)


def _outputs_dir(lecture_root: Path) -> Path:
    return lecture_root / "outputs"


def deep_dives_root(lecture_root: Path) -> Path:
    return _outputs_dir(lecture_root) / TOPIC_DEEP_DIVES_DIR


def deep_dive_markdown_path(lecture_root: Path, slug: str) -> Path:
    return deep_dives_root(lecture_root) / f"{slug}.md"


def deep_dive_index_path(lecture_root: Path) -> Path:
    return deep_dives_root(lecture_root) / INDEX_FILENAME


def _unique_slugs(topics: list[dict[str, Any]]) -> None:
    used: set[str] = set()
    for t in topics:
        base = slugify(t["title"])
        if not base:
            base = "topic"
        candidate = base
        n = 2
        while candidate in used:
            candidate = f"{base}-{n}"
            n += 1
        used.add(candidate)
        t["slug"] = candidate


def parse_topics_from_topic_map(md: str) -> list[dict[str, Any]]:
    """Parse ### headings and optional **Priorität:** / **Priority:** from 02_topic_map."""
    text = (md or "").strip()
    if not text:
        return []
    matches = list(_H3_TOPIC.finditer(text))
    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        pm = _PRIORITY.search(chunk)
        pr: int | None = int(pm.group(1)) if pm else None
        out.append({"title": title, "slug": "", "priority": pr})
    _unique_slugs(out)
    return out


def extract_topic_map_entry_block(topic_map_md: str, topic_title: str) -> str:
    """Return the markdown block for one ### topic (for prompt context)."""
    md = topic_map_md or ""
    esc = re.escape(topic_title.strip())
    m = re.search(rf"^###\s+{esc}\s*$", md, re.MULTILINE)
    if not m:
        return ""
    rest = md[m.end() :]
    nxt = re.search(r"^###\s+", rest, re.MULTILINE)
    block = rest[: nxt.start()] if nxt else rest
    block = block.strip()
    return block[:6000] + ("\n\n*[truncated]*\n" if len(block) > 6000 else "")


def extract_core_learning_section(core_md: str, topic_title: str) -> str:
    """Best-effort: slice ### lesson matching topic title."""
    md = core_md or ""
    esc = re.escape(topic_title.strip())
    m = re.search(rf"^###\s+{esc}\s*$", md, re.MULTILINE | re.IGNORECASE)
    if not m:
        # loose: first ### whose line normalizes similarly
        for mm in _H3_TOPIC.finditer(md):
            if mm.group(1).strip().lower() == topic_title.strip().lower():
                m = mm
                break
    if not m:
        return ""
    rest = md[m.end() :]
    nxt = re.search(r"^###\s+", rest, re.MULTILINE)
    block = rest[: nxt.start()] if nxt else rest
    block = block.strip()
    return block[:12000] + ("\n\n*[truncated]*\n" if len(block) > 12000 else "")


def load_topic_map_and_topics(lecture_root: Path) -> tuple[str | None, list[dict[str, Any]], str | None]:
    p, _ = resolve_existing_output(_outputs_dir(lecture_root), "topic_map")
    if p is None or not p.is_file():
        return None, [], "Topic roadmap file not found. Generate study materials first."
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, [], "Could not read topic roadmap."
    topics = parse_topics_from_topic_map(raw)
    return raw, topics, None


def _read_index(lecture_root: Path) -> dict[str, Any]:
    ip = deep_dive_index_path(lecture_root)
    if not ip.is_file():
        return {"version": 1, "entries": []}
    try:
        data = json.loads(ip.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "entries": []}
    if not isinstance(data, dict):
        return {"version": 1, "entries": []}
    entries = data.get("entries")
    if not isinstance(entries, list):
        data["entries"] = []
    return data


def _write_index(lecture_root: Path, data: dict[str, Any]) -> None:
    dr = deep_dives_root(lecture_root)
    dr.mkdir(parents=True, exist_ok=True)
    deep_dive_index_path(lecture_root).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _upsert_index_entry(
    lecture_root: Path,
    *,
    slug: str,
    title: str,
    priority: int | None,
) -> None:
    data = _read_index(lecture_root)
    entries: list[dict[str, Any]] = list(data.get("entries") or [])
    now = datetime.now(timezone.utc).isoformat()
    found = False
    for e in entries:
        if e.get("slug") == slug:
            e["title"] = title
            e["priority"] = priority
            e["generated_at"] = now
            e["updated_at"] = now
            found = True
            break
    if not found:
        entries.append(
            {
                "slug": slug,
                "title": title,
                "priority": priority,
                "generated_at": now,
                "updated_at": now,
            }
        )
    data["entries"] = entries
    _write_index(lecture_root, data)


def topic_entry_by_slug(topics: list[dict[str, Any]], slug: str) -> dict[str, Any] | None:
    for t in topics:
        if t.get("slug") == slug:
            return t
    return None


def deep_dive_exists(lecture_root: Path, slug: str) -> bool:
    p = deep_dive_markdown_path(lecture_root, slug)
    return p.is_file() and p.stat().st_size > 0


def read_deep_dive_markdown(lecture_root: Path, slug: str) -> str | None:
    p = deep_dive_markdown_path(lecture_root, slug)
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _truncate_for_prompt(full: str, cap: int) -> str:
    if len(full) <= cap:
        return full
    return full[:cap] + "\n\n---\n\n*[Truncated for generation.]*\n"


def _user_prompt_topic_deep_dive(
    *,
    analysis: LectureAnalysis,
    course_name: str,
    lecture_title: str,
    topic_title: str,
    priority: int | None,
    topic_map_excerpt: str,
    topic_lesson_excerpt: str,
    material_block: str,
) -> str:
    pri_line = f"Roadmap priority (if given): **{priority}/10**.\n\n" if priority is not None else ""

    if analysis.detected_language == "de":
        sections = (
            "Strukturiere die Ausgabe mit **genau** diesen ##-Abschnitten (Reihenfolge, deutsche Titel):\n\n"
            "## Warum dieses Thema in dieser Vorlesung zählt\n"
            "## Kernerklärung\n"
            "## Wichtige Unterscheidungen / typische Verwechslungen\n"
            "## Prüfungs- und Kursniveau\n"
            "## Beispiele\n"
            "## Aufgaben im Übungsstil\n"
            "## Erarbeitete Denkweise / Lösungslogik\n\n"
            "Inhaltliche Anforderungen:\n"
            "- Das ist die **Haupt-Lernseite** für genau dieses Thema — **deutlich tiefer** als die kurze Topic-Lektion "
            "auf der Vorlesungsseite, aber **kein** Lehrbuch-Kapitel und **keine** breite Fachfeld-Zusammenfassung.\n"
            "- **Nur** aus dem gelieferten Vorlesungs- und Übungsmaterial ableiten. Keine erfundenen Prüfungsfragen oder "
            "externe Theorie.\n"
            "- Tiefe: **so weit wie Vorlesung + Übungen + erkennbare Erwartungen im Material** rechtfertigen — nicht weiter.\n"
            "- **Kernerklärung:** klar, mit expliziten Unterscheidungen; treu zur Vorlesung.\n"
            "- **Prüfungs- und Kursniveau:** ehrlich aus der Quelle (Was scheint zentral? Was lohnt kein Überlernen?) — "
            "ohne konkrete Prüfungsgeheimnisse zu erfindern.\n"
            "- **Beispiele / Aufgaben / Lösungslogik:** wenn die Quelle Aufgaben oder Muster enthält, darauf abstimmen; "
            "sonst sparsam und sachlich.\n"
            "- Keine Meta-Floskeln („In diesem Abschnitt…“). Keine künstliche Länge.\n"
        )
        ctx = ""
        if topic_map_excerpt.strip():
            ctx += f"### Ausschnitt Topic Roadmap zu diesem Thema\n\n{topic_map_excerpt.strip()}\n\n"
        if topic_lesson_excerpt.strip():
            ctx += (
                "### Bestehende Topic-Lektion (Kontext — ersetzen durch tiefere Deep-Dive-Version)\n\n"
                f"{topic_lesson_excerpt.strip()}\n\n"
            )
        extra = (
            f"{sections}\n"
            f"{pri_line}"
            f"{_analysis_signal_lines(analysis)}\n"
            f"{_example_policy_line(analysis)}\n"
            f"{_artifact_technical_addon(analysis, 'core_learning')}\n"
            f"{_exercise_application_addon(analysis, 'core_learning')}\n\n"
            f"{ctx}"
            "### Zu bearbeitendes Thema\n\n"
            f"**Thema:** {topic_title}\n\n"
            "### Quellenmaterial (Vorlesung [+ Übungen])\n\n"
            f"{material_block}"
        )
    else:
        sections = (
            "Structure the output with **exactly** these ## sections (order, English titles):\n\n"
            "## Why this topic matters in this lecture\n"
            "## Core explanation\n"
            "## Important distinctions / what students confuse\n"
            "## Exam-level depth\n"
            "## Examples\n"
            "## Exercise-style tasks\n"
            "## Worked reasoning / solution logic\n\n"
            "Requirements:\n"
            "- This is the **main study page** for this single topic — **deeper** than the short Topic Lesson on the "
            "lecture page, but **not** a textbook chapter and **not** a broad field survey.\n"
            "- **Only** ground claims in the lecture + exercise material provided. No invented exam specifics or "
            "external theory.\n"
            "- Depth: **as far as lecture + exercises + evidence in the material** justify — not beyond.\n"
            "- **Exam-level depth:** be honest from the source (what seems central vs not worth overlearning) — "
            "do not invent secret exam knowledge.\n"
            "- **Examples / tasks / reasoning:** align with worksheet patterns when present.\n"
            "- No filler meta-narration. No padding.\n"
        )
        ctx = ""
        if topic_map_excerpt.strip():
            ctx += f"### Topic roadmap excerpt for this topic\n\n{topic_map_excerpt.strip()}\n\n"
        if topic_lesson_excerpt.strip():
            ctx += (
                "### Existing topic lesson (context — supersede with a deeper deep dive)\n\n"
                f"{topic_lesson_excerpt.strip()}\n\n"
            )
        extra = (
            f"{sections}\n"
            f"{pri_line}"
            f"{_analysis_signal_lines(analysis)}\n"
            f"{_example_policy_line(analysis)}\n"
            f"{_artifact_technical_addon(analysis, 'core_learning')}\n"
            f"{_exercise_application_addon(analysis, 'core_learning')}\n\n"
            f"{ctx}"
            "### Topic to develop\n\n"
            f"**Topic:** {topic_title}\n\n"
            "### Source material (lecture [+ exercises])\n\n"
            f"{material_block}"
        )
    return extra.strip()


def run_topic_deep_dive_generation(lecture_id: int, slug: str) -> tuple[bool, str]:
    """Generate and save one deep dive. Returns (ok, message)."""
    if not openai_service.is_openai_configured():
        return False, "OpenAI is not configured. Set OPENAI_API_KEY in your .env file."

    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return False, "Lecture not found."

    prep = prepare_generation_inputs(lecture_id)
    if not prep.ok or not prep.payload:
        return False, prep.reason

    lecture_root = lecture_root_from_source_relative(lec["source_file_path"])
    topic_map_md, topics, err = load_topic_map_and_topics(lecture_root)
    if err:
        return False, err
    entry = topic_entry_by_slug(topics, slug)
    if not entry:
        return False, "Unknown topic slug."

    topic_title = entry["title"]
    priority = entry.get("priority")
    if isinstance(priority, int) and not (1 <= priority <= 10):
        priority = None

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
    material_block = _truncate_for_prompt(material_block, 110_000)

    tm_ex = extract_topic_map_entry_block(topic_map_md or "", topic_title) if topic_map_md else ""
    core_path, _ = resolve_existing_output(_outputs_dir(lecture_root), "core_learning")
    lesson_ex = ""
    if core_path and core_path.is_file():
        try:
            core_raw = core_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            core_raw = ""
        lesson_ex = extract_core_learning_section(core_raw, topic_title)

    sys = _system_prompt(analysis) + "\n\n" + (
        "**Topic Deep Dive (zusätzlich):** Genau **ein** Thema. Längere Ausarbeitung als die Topic-Lektion, "
        "aber immer noch quellgebunden und prüfungsrealistisch begrenzt.\n"
        if analysis.detected_language == "de"
        else "**Topic Deep Dive (add-on):** Exactly **one** topic. Longer than the Topic Lesson, "
        "still source-bound and realistically exam-scoped.\n"
    )

    user = _user_prompt_topic_deep_dive(
        analysis=analysis,
        course_name=prep.payload["course_name"],
        lecture_title=prep.payload["lecture_title"],
        topic_title=topic_title,
        priority=priority,
        topic_map_excerpt=tm_ex,
        topic_lesson_excerpt=lesson_ex,
        material_block=material_block,
    )

    ok, md_out, err_msg = openai_service.chat_completion_markdown(
        system_prompt=sys,
        user_prompt=user,
        max_tokens=8192,
    )
    if not ok:
        return False, err_msg or "Generation failed."

    md_out = cleanup_generated_markdown(md_out)
    if not md_out.strip():
        return False, "Model returned empty text."

    dr = deep_dives_root(lecture_root)
    dr.mkdir(parents=True, exist_ok=True)
    deep_dive_markdown_path(lecture_root, slug).write_text(md_out.strip() + "\n", encoding="utf-8")
    _upsert_index_entry(lecture_root, slug=slug, title=topic_title, priority=priority)
    return True, f"Topic deep dive saved for “{topic_title}”."


def build_lecture_page_context(lecture_id: int) -> dict[str, Any]:
    """Template context: topics list, slug map JSON, optional error."""
    lec = lecture_service.get_lecture_by_id(lecture_id)
    if not lec:
        return {"topics": [], "slug_by_title": {}, "error": "Lecture not found.", "has_topic_map": False}

    try:
        root = lecture_root_from_source_relative(lec["source_file_path"])
    except (OSError, ValueError):
        return {"topics": [], "slug_by_title": {}, "error": "Invalid lecture path.", "has_topic_map": False}

    _tm, topics, err = load_topic_map_and_topics(root)
    if err:
        return {"topics": [], "slug_by_title": {}, "error": err, "has_topic_map": False}

    slug_by_title: dict[str, str] = {}
    enriched: list[dict[str, Any]] = []
    for t in topics:
        title = t["title"]
        slug = t["slug"]
        slug_by_title[title] = slug
        pr = t.get("priority")
        enriched.append(
            {
                "title": title,
                "slug": slug,
                "priority": pr,
                "has_deep_dive": deep_dive_exists(root, slug),
                "recommended": isinstance(pr, int) and pr >= RECOMMENDED_PRIORITY_MIN,
            }
        )

    return {
        "topics": enriched,
        "slug_by_title": slug_by_title,
        "error": None,
        "has_topic_map": True,
    }


def list_missing_recommended_deep_dives(limit: int = 12) -> list[dict[str, Any]]:
    """
    High-priority roadmap topics without a generated deep dive yet (for planner hints).
    """
    rows = lecture_service.list_lectures_for_planner()
    out: list[dict[str, Any]] = []
    for lec in rows:
        if lec.get("status") != "generation_complete":
            continue
        try:
            root = lecture_root_from_source_relative(lec["source_file_path"])
        except (OSError, ValueError):
            continue
        _tm, topics, err = load_topic_map_and_topics(root)
        if err or not topics:
            continue
        for t in topics:
            pr = t.get("priority")
            if not isinstance(pr, int) or pr < RECOMMENDED_PRIORITY_MIN:
                continue
            if deep_dive_exists(root, t["slug"]):
                continue
            out.append(
                {
                    "lecture_id": lec["id"],
                    "lecture_title": lec["title"],
                    "course_name": lec["course_name"],
                    "topic_title": t["title"],
                    "slug": t["slug"],
                    "priority": pr,
                }
            )
            if len(out) >= limit:
                return out
    return out
