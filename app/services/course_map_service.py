"""Course map + recommended study order from early lectures (TOC), titles, and concept overlap."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import APP_ROOT
from app.db.database import get_connection
from app.services.lecture_paths import lecture_root_from_source_relative
from app.services.topic_deep_dive import load_topic_map_and_topics

_EARLY_LECTURE_COUNT = 4
_READ_HEAD_CHARS = 55_000
_TOC_WINDOW_CHARS = 14_000

_TOC_ANCHOR = re.compile(
    r"(?is)"
    r"(?:^|\n)\s*(inhaltsverzeichnis|table\s+of\s+contents|contents|"
    r"inhalt|überblick|uberblick|course\s+outline|semesterplan|"
    r"organisatorischer\s+überblick|organisatorischer\s+uberblick)\b[^\n]{0,120}\n"
)

_NUMBERED_LINE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s*[.)]\s*(?P<title>.{4,200})\s*$",
    re.MULTILINE,
)
_LECTURE_LINE = re.compile(
    r"^\s*(?:vorlesung|lecture|lec\.?|unit|kapitel|teil|session)\s*(?P<num>\d{1,2})\s*[:\).\-\u2013\u2014]\s*(?P<title>.{3,200})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_INTRO_HINTS = re.compile(
    r"organisator|überblick|uberblick|einführung|einfuhrung|introduction|overview|"
    r"inhaltsverzeichnis|motivation|ziele|course\s+outline",
    re.IGNORECASE,
)
_APPLIED_HINTS = re.compile(
    r"übung|ubung|exercise|anwendung|application|praxis|projekt|workshop|case\s+study",
    re.IGNORECASE,
)


def _normalize(s: str) -> str:
    t = (s or "").lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("ß", "ss")
    t = re.sub(r"[^a-z0-9äöü]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(s: str) -> set[str]:
    return {w for w in _normalize(s).split() if len(w) > 2}


def _lectures_ordered(course_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT id, title, slug, study_progress, status, source_file_path, extracted_text_path, created_at
            FROM lectures
            WHERE course_id = ?
            ORDER BY id ASC
            """,
            (course_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def _read_extracted_head(lec: dict[str, Any], max_chars: int = _READ_HEAD_CHARS) -> str:
    rel = (lec.get("extracted_text_path") or "").strip().replace("\\", "/")
    if not rel:
        root = _lecture_root(lec)
        if root is None:
            return ""
        p = root / "extracted_text.txt"
    else:
        p = (APP_ROOT / rel).resolve()
    if not p.is_file():
        return ""
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return raw[:max_chars]


def _lecture_root(lec: dict[str, Any]) -> Path | None:
    sp = (lec.get("source_file_path") or "").strip()
    if not sp:
        return None
    try:
        return lecture_root_from_source_relative(sp)
    except (OSError, ValueError):
        return None


def _toc_window_from_combined_text(text: str) -> str:
    if not text.strip():
        return ""
    m = _TOC_ANCHOR.search(text)
    if m:
        start = m.start()
        return text[start : start + _TOC_WINDOW_CHARS]
    # No explicit TOC heading: use head (often intro slides list outline)
    return text[:_TOC_WINDOW_CHARS]


def _parse_toc_entries(window: str) -> list[tuple[int | None, str]]:
    entries: list[tuple[int | None, str]] = []
    seen: set[str] = set()
    for m in _NUMBERED_LINE.finditer(window):
        num = int(m.group("num"))
        title = m.group("title").strip()
        key = _normalize(title)
        if len(key) < 6 or key in seen:
            continue
        seen.add(key)
        entries.append((num, title))
    for m in _LECTURE_LINE.finditer(window):
        num = int(m.group("num"))
        title = m.group("title").strip()
        key = _normalize(title)
        if len(key) < 6 or key in seen:
            continue
        seen.add(key)
        entries.append((num, title))
    return entries[:40]


def _match_toc_to_lectures(
    lectures: list[dict[str, Any]], toc_entries: list[tuple[int | None, str]]
) -> list[int]:
    """Return lecture ids in TOC-suggested order (best matches only)."""
    ordered: list[int] = []
    used: set[int] = set()
    for num, toc_title in toc_entries:
        best_id: int | None = None
        best_score = 0.0
        ttoks = _tokens(toc_title)
        for lec in lectures:
            lid = int(lec["id"])
            if lid in used:
                continue
            title = str(lec.get("title") or "")
            toks = _tokens(title)
            overlap = len(ttoks & toks)
            score = float(overlap) * 3.1
            if num is not None:
                if re.search(rf"\b0*{num}\b", _normalize(title)):
                    score += 8.0
            if score > best_score:
                best_score = score
                best_id = lid
        if best_id is not None and best_score >= 2.0:
            ordered.append(best_id)
            used.add(best_id)
    return ordered


def _concept_edges(course_id: int, min_shared: int = 2) -> list[tuple[int, int, int]]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT lc1.lecture_id AS a, lc2.lecture_id AS b, COUNT(*) AS shared
            FROM lecture_concepts lc1
            JOIN lecture_concepts lc2
              ON lc1.concept_id = lc2.concept_id AND lc1.lecture_id < lc2.lecture_id
            JOIN lectures l1 ON l1.id = lc1.lecture_id
            JOIN lectures l2 ON l2.id = lc2.lecture_id
            WHERE l1.course_id = ? AND l2.course_id = ?
            GROUP BY lc1.lecture_id, lc2.lecture_id
            HAVING COUNT(*) >= ?
            """,
            (course_id, course_id, min_shared),
        )
        return [(int(r["a"]), int(r["b"]), int(r["shared"])) for r in cur.fetchall()]


def _clusters_for_lectures(
    lecture_ids: list[int], edges: list[tuple[int, int, int]]
) -> dict[int, int]:
    """Map lecture_id -> cluster index (0-based)."""
    if not lecture_ids:
        return {}
    parent: dict[int, int] = {i: i for i in lecture_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _ in edges:
        if a in parent and b in parent:
            union(a, b)
    roots: dict[int, list[int]] = defaultdict(list)
    for lid in lecture_ids:
        roots[find(lid)].append(lid)
    clusters = sorted(roots.values(), key=lambda xs: min(xs))
    out: dict[int, int] = {}
    for idx, group in enumerate(clusters):
        for lid in group:
            out[lid] = idx
    return out


def _role_for_lecture(idx: int, n: int, title: str) -> str:
    t = title or ""
    if _INTRO_HINTS.search(t) or (idx == 0 and n > 1):
        return "foundation"
    if _APPLIED_HINTS.search(t):
        return "applied"
    if idx < max(2, n // 4):
        return "foundation"
    if idx >= n - max(1, n // 5):
        return "applied"
    return "core"


def _build_study_order(
    lectures: list[dict[str, Any]],
    toc_order: list[int],
    edges: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    by_id = {int(l["id"]): l for l in lectures}
    seq_ids = [int(l["id"]) for l in lectures]
    # Merge: TOC order first, then remaining in sequence order
    ordered: list[int] = []
    seen: set[int] = set()
    for lid in toc_order:
        if lid in by_id and lid not in seen:
            ordered.append(lid)
            seen.add(lid)
    for lid in seq_ids:
        if lid not in seen:
            ordered.append(lid)
            seen.add(lid)
    steps: list[dict[str, Any]] = []
    strongest_prev: dict[int, tuple[int, int]] = {}
    for a, b, sh in sorted(edges, key=lambda x: -x[2]):
        if b not in strongest_prev or sh > strongest_prev[b][1]:
            strongest_prev[b] = (a, sh)

    for i, lid in enumerate(ordered):
        lec = by_id[lid]
        note_parts: list[str] = []
        if i == 0:
            note_parts.append("Start here")
        if strongest_prev.get(lid):
            pa, sh = strongest_prev[lid]
            if pa in by_id:
                prev_i = ordered.index(pa) if pa in ordered else -1
                if prev_i >= 0 and prev_i < i:
                    note_parts.append(f"after «{by_id[pa]['title'][:44]}» ({sh} shared topics)")
                elif prev_i > i:
                    note_parts.append(f"topics overlap «{by_id[pa]['title'][:44]}» — consider reviewing that one first")
        if toc_order and lid == toc_order[0] and i == 0 and len(toc_order) >= 2:
            note_parts.append("aligned with outline in early lecture(s)")
        steps.append(
            {
                "lecture_id": lid,
                "title": lec["title"],
                "href": f"/lectures/{lid}",
                "study_progress": lec.get("study_progress") or "not_started",
                "note": " · ".join(dict.fromkeys(note_parts)) if note_parts else "",
            }
        )
    return steps


def _structure_bullets(
    lectures: list[dict[str, Any]],
    toc_order: list[int],
    clusters: dict[int, int],
    edges: list[tuple[int, int, int]],
) -> list[str]:
    bullets: list[str] = []
    n = len(lectures)
    if n == 0:
        return bullets
    if toc_order:
        bullets.append(
            f"Outline in early lecture(s) suggests an order for {len(toc_order)} lecture(s) — study path uses that as anchor."
        )
    if clusters:
        n_clusters = len(set(clusters.values()))
        if n_clusters > 1:
            bullets.append(f"{n_clusters} topic clusters from shared concepts — lectures in a cluster reinforce each other.")
    if edges:
        top = max(edges, key=lambda x: x[2])
        a, b, sh = top
        ta = next((l["title"] for l in lectures if int(l["id"]) == a), "?")
        tb = next((l["title"] for l in lectures if int(l["id"]) == b), "?")
        bullets.append(f"Strongest concept bridge: «{ta[:40]}» ↔ «{tb[:40]}» ({sh} shared topics).")
    if n >= 3:
        bullets.append(
            f"First lectures ({min(3, n)} in sequence) anchor structure; later lectures often build on vocabulary from those."
        )
    return bullets[:6]


def build_course_map_and_path(course_id: int) -> dict[str, Any]:
    """
    Deterministic course map + study path.

    Anchors on combined extracted text from the first few lectures (TOC / overview),
    then titles, sequence, and concept overlap.
    """
    lectures = _lectures_ordered(course_id)
    if not lectures:
        return {
            "lectures": [],
            "anchors_scanned": [],
            "toc_entries": [],
            "toc_order_ids": [],
            "edges": [],
            "clusters": {},
            "cluster_labels": [],
            "map_nodes": [],
            "structure_bullets": [],
            "study_steps": [],
            "disclaimer": "No lectures in this course yet.",
        }

    combined_early = ""
    anchors_scanned: list[dict[str, Any]] = []
    for lec in lectures[:_EARLY_LECTURE_COUNT]:
        head = _read_extracted_head(lec)
        if head.strip():
            anchors_scanned.append(
                {
                    "id": int(lec["id"]),
                    "title": lec["title"],
                    "chars": len(head),
                }
            )
            combined_early += f"\n\n---\n## Lecture excerpt: {lec['title']}\n\n" + head
        # Topic map headings (generated roadmap) strengthen structure when TOC is thin
        sp = (lec.get("source_file_path") or "").strip()
        if sp and str(lec.get("status") or "") == "generation_complete":
            try:
                root = lecture_root_from_source_relative(sp)
                _tm, topics, err = load_topic_map_and_topics(root)
                if topics and not err:
                    lines = "\n".join(f"{i + 1}. {t.get('title') or ''}" for i, t in enumerate(topics[:28]))
                    combined_early += f"\n\n---\n## Topic map headings: {lec['title']}\n\n{lines}\n"
            except (OSError, ValueError):
                pass

    window = _toc_window_from_combined_text(combined_early)
    toc_entries = _parse_toc_entries(window)
    toc_order = _match_toc_to_lectures(lectures, toc_entries)
    edges = _concept_edges(course_id, min_shared=2)
    lecture_ids = [int(l["id"]) for l in lectures]
    clusters = _clusters_for_lectures(lecture_ids, edges)
    cluster_sizes: dict[int, int] = defaultdict(int)
    for lid in lecture_ids:
        cluster_sizes[clusters.get(lid, 0)] += 1
    cluster_labels = [
        {"cluster_id": cid, "size": sz, "label": f"Cluster {cid + 1}"}
        for cid, sz in sorted(cluster_sizes.items(), key=lambda x: x[0])
    ]

    n = len(lectures)
    map_nodes: list[dict[str, Any]] = []
    for idx, lec in enumerate(lectures):
        lid = int(lec["id"])
        map_nodes.append(
            {
                "lecture_id": lid,
                "title": lec["title"],
                "href": f"/lectures/{lid}",
                "seq": idx + 1,
                "role": _role_for_lecture(idx, n, str(lec.get("title") or "")),
                "cluster_id": clusters.get(lid, 0),
                "study_progress": lec.get("study_progress") or "not_started",
                "in_toc_order": lid in toc_order,
            }
        )

    study_steps = _build_study_order(lectures, toc_order, edges)
    bullets = _structure_bullets(lectures, toc_order, clusters, edges)

    disclaimer = (
        "Based on early lecture text (when available), titles, sequence, and indexed concepts — "
        "not a perfect syllabus graph."
    )
    if not anchors_scanned:
        disclaimer += " No extracted text found on the first lectures yet — map follows upload order and concepts only."

    return {
        "lectures": lectures,
        "anchors_scanned": anchors_scanned,
        "toc_entries": [{"num": n, "title": t} for n, t in toc_entries[:20]],
        "toc_order_ids": toc_order,
        "edges": [{"a": a, "b": b, "shared": sh} for a, b, sh in edges[:24]],
        "clusters": clusters,
        "cluster_labels": cluster_labels,
        "map_nodes": map_nodes,
        "structure_bullets": bullets,
        "study_steps": study_steps,
        "disclaimer": disclaimer,
    }
