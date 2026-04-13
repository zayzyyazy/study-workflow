# Study Workflow App

Local web app for organizing university courses and lecture files. Built with FastAPI, Jinja2 templates, and SQLite. Runs entirely on your machine — no cloud, no accounts required (OpenAI key optional for AI generation).

A separate Electron desktop wrapper exists to run this as a native window. See [Electron wrapper](#electron-desktop-wrapper) below.

---

## What it does

- **Course dashboard** — home page organized by course
- **Lecture folder view** — each course lists lectures in folder-style groups
- **Lecture study workspace** — per-lecture page with all study outputs, progress tracking, and actions
- **Upload lecture files** — PDF, DOCX, TXT, MD; text is extracted automatically
- **Generate study materials** — OpenAI-powered, produces 5 structured outputs + a combined study pack
- **Rebuild combined study pack** — reassembles the study pack from existing files without re-running OpenAI
- **Printable study pack** — dedicated HTML view → browser print → Save as PDF
- **Download study_pack.md** — download the combined markdown file
- **Math rendering** — KaTeX support in the lecture workspace
- **Study progress tracking** — mark lectures as `not_started`, `in_progress`, or `done`
- **Star / priority lectures** — starred lectures appear in a dedicated section on the home page
- **Concept indexing** — key terms extracted from generated files, listed on the course page per lecture
- **Search and filters** — global lecture search on home; filter by title/status on course page
- **ZIP export** — download a single lecture folder or entire course as a ZIP
- **Bulk generation** — generate all ready lectures in a course in one click
- **Safe reset tools** — database-level reset helpers for testing/development
- **Storage visibility** — folder/storage info visible in the workspace

---

## Study outputs

After generation, each lecture has:

| File | Section |
|------|---------|
| `01_glossary.md` | Glossary |
| `02_teach_me.md` | Teach Me |
| `03_worked_examples.md` | Worked Examples |
| `04_mistakes_and_checks.md` | Mistakes and Checks |
| `05_revision_sheet.md` | Revision Sheet |
| `06_study_pack.md` | Combined Study Pack (all sections, no LLM) |

---

## Lecture statuses

Pipeline statuses (set automatically):

`uploaded` → `text_extracted` / `extraction_failed` → `ready_for_generation` → `generation_pending` → `generation_complete` / `generation_failed`

Study progress (set by you): `not_started` · `in_progress` · `done`

---

## Setup

### 1. Python

Python 3.10+ (3.11 or 3.12 recommended).

### 2. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

```bash
cp .env.example .env
```

Edit `.env`:

- **Paths** — defaults: SQLite DB at `data/app.db`, lecture files under `courses/`.
- **OpenAI (optional)** — needed only for **Generate study materials**:
  - `OPENAI_API_KEY` — from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - `OPENAI_MODEL` — defaults to `gpt-4o-mini`

The app runs without an API key. Generation buttons return an error until you set the key.

### 5. Run

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Where files are stored

| What | Location |
|------|----------|
| SQLite database | `data/app.db` |
| Uploaded lectures | `courses/<course-slug>/Lecture NN - <title>/source/` |
| Per-lecture metadata | `.../meta.json` |
| Extracted text | `.../extracted_text.txt` |
| Study outputs | `.../outputs/01_glossary.md` … `06_study_pack.md` |
| Course concept index | `courses/<course-slug>/course_index/concept_index.md` |

---

## Project structure

```
study_workflow_app/
├── main.py                  # FastAPI entry point
├── requirements.txt
├── .env.example
├── app/
│   ├── routes/              # HTTP route handlers (home, courses, lectures, upload)
│   ├── services/            # Business logic (generation, extraction, export, etc.)
│   ├── templates/           # Jinja2 HTML templates
│   ├── static/              # CSS, JS, KaTeX vendor files
│   └── db/                  # SQLite connection + schema
├── courses/                 # Runtime: lecture files (git-ignored)
└── data/                    # Runtime: SQLite DB (git-ignored)
```

---

## Electron desktop wrapper

A separate Electron repo wraps this app in a native desktop window. It spawns the local Python/uvicorn server and opens it in a window.

To run the Electron wrapper locally:

1. Make sure the Python app runs at `http://127.0.0.1:8000`.
2. Clone or open the Electron wrapper repo (separate from this folder).
3. Update the local Python path in the Electron config to match your machine.
4. `npm install` then `npm start`.

> The Electron wrapper is not in this repository. Do not confuse the two.

---

## Known limitations

- **PDF export** — the app produces a printable HTML page. PDF is created via browser print → Save as PDF. There is no server-side PDF generation.
- **Rebuild study pack** — reassembles sections from existing files without re-calling OpenAI. It does not regenerate content.
- **Old lectures** — lectures generated before the study-pack redesign (v1 output names) may need regeneration to fully use the new structure. Legacy filenames are tried as fallbacks.
- **Concept extraction** — deterministic parsing of headings and glossary lines, not semantic/NLP. Similar phrases with different wording may appear separately; noisy bold text can slip through.
- **Language detection** — heuristic German vs English only, not a separate classifier. Short or mixed-language slides may be misclassified.
- **Generation is synchronous** — no queue or background workers. Bulk generate waits in the browser until all runs finish.
- **Search** — substring match only, no ranking or FTS.
- **Electron path** — the desktop wrapper requires the local Python app path to be correct for your machine.
- **OpenAI costs** — generation makes 5 API calls per lecture. Costs depend on your plan and model.

---

## What's next (rough ideas)

- SQLite FTS5 for faster full-text search
- Background job queue for generation (instead of blocking HTTP)
- Scheduled revision reminders / calendar export
- Cross-lecture concept linking
