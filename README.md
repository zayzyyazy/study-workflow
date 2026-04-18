# Study Workflow App (Study AI)

Local-first web app for university lectures: upload PDFs (and other sources), organize by course, extract text, generate AI study materials, and drill down with **topic deep dives**, **practice questions**, and **subtopic deep dives**. Includes **study progress**, **stars**, a **schedule-aware planner**, and a **home dashboard**. Runs entirely on your machine; an optional **OpenAI** key powers generation.

A separate **Electron** desktop wrapper (`studyai`) can launch the same FastAPI server in a native window — it is **not** in this repository.

---

## What this project is

- **FastAPI** + **Jinja2** + **SQLite** + on-disk files under `courses/`.
- No accounts, no cloud storage — data stays on your computer.
- **Optional AI**: study materials, topic deep dives, example questions, and subtopic deep dives call the OpenAI API when `OPENAI_API_KEY` is set.

---

## Core features (what they’re for)

| Area | Purpose |
|------|--------|
| **Courses & lectures** | Group uploads by course. Each lecture is a folder with sources, extracted text, and generated outputs. |
| **Upload & extraction** | Add PDFs/DOCX/TXT/MD; text is extracted for search and generation. **Multi-source** lectures can attach Übungsblätter with roles (lecture vs exercise) so generation can weight lecture vs sheet. |
| **Smart titles (new uploads)** | Display titles use `Lecture NN - …` with a **readable topic** inferred from the **start of extracted text** (headings, “Vorlesung N: …”, etc.) when possible; folder names stay stable from the initial name. |
| **Study material generation** | One pipeline produces structured Markdown: quick overview, topic roadmap, topic lessons (core learning), revision sheet, and a **combined study pack** (reassembled from sections without extra AI). |
| **Topic roadmap / topic lessons** | Roadmap lists prioritized topics; topic lessons expand each topic at lecture-page level. |
| **Topic deep dives** | Per roadmap topic, an on-demand **longer** page (still source-bound) for real studying — not a second thin summary. |
| **Example questions** | From a topic deep dive page: generate **easy / medium / hard** question sets (Markdown files). Style follows lecture + exercises; “hard” means **exam-level for this course**, not abstract competition problems. |
| **Subtopic deep dives** | If the topic deep dive uses `##` sections, each section can link to a **narrower** deep dive for that subtopic only (on-demand Markdown). |
| **Planner** | Weekly or one-off schedule blocks (lecture / project / deadline). **Deterministic** suggestions: what to do today, catch-up, next class, missing high-priority deep dives — tied to **your** lectures and progress (no separate AI coach). |
| **Dashboard (home)** | Continue / not started picks, planner “next”, suggested focus, courses needing work, deep-dive gaps, recent lectures — compact, not a second planner. |
| **Progress & stars** | Per lecture: `not_started` · `in_progress` · `done`. Star important lectures for quick access on the home page. |
| **Print & export** | Printable HTML for the study pack; download `study_pack.md`; **ZIP** export of a lecture folder or whole course (includes generated files and topic deep dive extras). |
| **Concept index** | Deterministic term/heading extraction into a per-course concept list. |
| **Electron wrapper** | Separate repo starts `uvicorn` and opens `http://127.0.0.1:8000` in a window. See [Electron wrapper](#electron-desktop-wrapper) below. |

---

## Typical study flow

1. **Create or pick a course** → **Upload** a lecture file (and optionally add exercise PDFs with the exercise role).
2. **Re-run extraction** if needed → when status allows, **Generate study materials** (requires API key).
3. On the **lecture page**, read the overview and topic roadmap / lessons for orientation.
4. Open a **topic deep dive** for a roadmap topic you care about; **regenerate** if you want a fresher pass.
5. From that page: **Generate example questions** (pick difficulty) to practice; open **subtopic deep dives** from `##` sections when a subsection is still unclear.
6. Set **study progress** and **star** lectures you’re actively working on.
7. Add **planner** blocks (link lecture slots to a course when relevant) and use **Home** + **Planner** to decide **today / tomorrow** and what’s still missing (e.g. recommended deep dives).

---

## Project structure (high level)

```
study_workflow_app/
├── main.py                 # FastAPI entry
├── requirements.txt
├── .env.example            # copy to .env
├── app/
│   ├── routes/             # home, courses, lectures, upload, planner
│   ├── services/           # extraction, generation, topic deep dives, planner, export, …
│   ├── templates/          # Jinja2 HTML
│   ├── static/             # CSS, JS, KaTeX
│   └── db/                 # SQLite connection + schema migrations on startup
├── courses/                # runtime: all lecture/course files (usually git-ignored)
└── data/                   # runtime: SQLite DB (usually git-ignored)
```

---

## Setup

### Requirements

- **Python 3.10+** (3.11+ recommended).

### Virtual environment

```bash
cd /path/to/study_workflow_app
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` as needed:

| Variable | Role |
|----------|------|
| `DATABASE_PATH` | SQLite file (default `./data/app.db`) |
| `COURSES_STORAGE_DIR` | Root for lecture folders (default `./courses`) |
| `APP_DATA_DIR` | Data directory (default `./data`) |
| `OPENAI_API_KEY` | Required for any AI generation |
| `OPENAI_MODEL` | e.g. `gpt-4o-mini` (default) |
| `GENERATION_MODE` | `strict_v2` (default) or `legacy` — prompt discipline for main study materials |

The app **starts** without an API key; generation actions show a clear error until the key is set.

### Run the web app

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Study outputs (current default filenames)

After generation, under each lecture’s `outputs/` you’ll typically see:

| File | Role |
|------|------|
| `01_quick_overview.md` | Short orientation |
| `02_topic_map.md` | Topic roadmap (priorities) |
| `03_core_learning.md` | Topic lessons |
| `04_revision_sheet.md` | Compact revision |
| `05_study_pack.md` | Combined pack (built from the sections above) |

Older lectures may still resolve **legacy** filenames (e.g. older teach-me names); the app tries fallbacks when loading.

**Topic-level extras** (same lecture folder, `outputs/topic_deep_dives/`):

| Pattern | Role |
|---------|------|
| `{topic-slug}.md` | Main topic deep dive |
| `{topic-slug}_questions_easy.md` / `_medium` / `_hard` | Example questions |
| `{topic-slug}_sub_{subslug}.md` | Subtopic deep dive |
| `index.json` | Small metadata index for deep dives |

---

## Lecture pipeline vs your study marks

**Pipeline** (automatic): `uploaded` → extraction → `ready_for_generation` → `generation_complete` / failures (see UI).

**Study progress** (you): `not_started` · `in_progress` · `done` — independent of generation status.

---

## Storage & data layout

| What | Where |
|------|--------|
| SQLite DB | `data/app.db` by default — courses, lectures, artifacts rows, concepts, **planner schedule items** (`planner_schedule_items`), etc. |
| Course folders | `courses/<course-slug>/` |
| Each lecture | `courses/.../Lecture NN - <title>/` |
| Sources | `.../source/` (uploaded files) |
| Manifest | `meta.json` at lecture root |
| Extracted text | `extracted_text.txt` (combined sources) |
| Generated study sections | `.../outputs/*.md` |
| Topic deep dives & extras | `.../outputs/topic_deep_dives/` (see table above) |
| Course concept index | `courses/<course-slug>/course_index/` |

**Planner data** lives only in SQLite (`planner_schedule_items`), not in separate files.

---

## Electron desktop wrapper

The wrapper lives in a **different** clone, e.g.:

- `~/Desktop/Projects/studyai`

It expects the FastAPI project path in **`main/main.js`** (`PYTHON_APP_ROOT`), for example:

`/Users/you/Desktop/Projects/Study-bot/study_workflow_app`

From the Electron folder:

```bash
cd /path/to/studyai
npm install
npm run dev          # or npm start — opens Electron, may spawn uvicorn
npm run dist         # packaged macOS build (see studyai/package.json)
```

Ensure this repo has a `.venv` with dependencies installed; Electron looks for `python` under `.venv` or `venv`.

---

## Other useful actions

- **Bulk generate** — course page can generate all lectures that are ready (sequential; waits until done).
- **Rebuild study pack** — recombine `05_study_pack.md` from existing section files without AI.
- **ZIP export** — lecture or whole course folder including `outputs/` and `topic_deep_dives/`.

---

## Limitations (honest)

- **Local-only** — no sync, no cloud backup, no multi-device by default.
- **No Google Calendar sync**, **no push notifications**, **no full calendar UI** — planner is schedule blocks + deterministic hints.
- **No automated grading** — example questions are Markdown for self-study; no quiz engine or answer checking.
- **Quality follows sources** — messy PDFs, scans, or tiny extracts limit titles, roadmap, and deep dives.
- **Generation is synchronous** — long runs block until the HTTP request finishes (bulk included).
- **Search** — simple substring search, not full-text search indexing.
- **Language** — heuristics favor DE/EN; mixed or very short text can be misclassified.
- **Costs** — each generation uses the OpenAI API; deep dives and question sets are separate calls from the main five study files.
- **Electron** — path and port must match your machine (`127.0.0.1:8000` by default in the wrapper).
