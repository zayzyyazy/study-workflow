# Study Workflow App (C-lite)

Local web app for organizing university courses and lecture files on your laptop. It stores uploads on disk, extracts text, and can **generate study materials** (glossary, summary, topics, deep dive, connections) via OpenAI.

## Setup

### 1. Python

Use Python 3.10+ (3.11 or 3.12 recommended).

### 2. Virtual environment

From the project root (`study_workflow_app/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copy the example env file:

```bash
cp .env.example .env
```

Edit `.env`:

- **Paths** — defaults store the SQLite DB under `data/` and lecture files under `courses/`.
- **OpenAI (optional)** — required only if you use **Generate study materials**:
  - `OPENAI_API_KEY` — from [OpenAI API keys](https://platform.openai.com/api-keys)
  - `OPENAI_MODEL` — defaults to `gpt-4o-mini` (change if your account uses another model name)

The app runs **without** an API key; generation buttons show an error until you set the key.

### 5. Run

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Where files are saved

| What | Location |
|------|-----------|
| SQLite database | `data/app.db` (or `DATABASE_PATH` in `.env`) |
| Uploaded lectures | `courses/<course-slug>/Lecture NN - <title>/source/` |
| Per-lecture metadata | `.../meta.json` |
| Extracted plain text | `.../extracted_text.txt` (when extraction succeeds) |
| Generated study materials | `.../outputs/01_glossary.md` … `05_connections.md` |
| Course concept index (Markdown) | `courses/<course-slug>/course_index/concept_index.md` |

## Concept indexing (course level)

After **study materials are generated successfully**, the app **automatically** extracts key terms and topics **without extra LLM calls**:

- **Sources**: `01_glossary.md`, `02_summary.md`, `03_topic_explanations.md`, `04_deep_dive.md` (not connections).
- **Method**: deterministic parsing — list lines and tables in the glossary, `##` / `###` headings, and `**bold**` phrases (capped), plus normalization for deduplication.
- **Database**: rows in **`concepts`** (unique `normalized_name`) and **`lecture_concepts`** (per lecture). Re-running generation **replaces** links for that lecture so duplicates do not accumulate.
- **UI**: the **course** page lists concepts sorted by how many lectures mention them, with links to each lecture. The **lecture** page shows a compact tag list when concepts exist.

**Limitations**: This is not semantic search — similar phrases with different wording may appear twice. Very generic headings may be filtered; noisy bold text can appear occasionally. No cross-course linking.

## How generation works

1. Upload a lecture and wait until status is **`ready_for_generation`** (extraction succeeded).
2. Open the lecture page and click **Generate study materials**.
3. Status moves to **`generation_pending`**, then **`generation_complete`** on success, or **`generation_failed`** if something goes wrong.
4. Before each generation run, the app **analyzes the extracted text** (no extra LLM call): **German vs English** (`de` / `en`) and a **content profile** — **general**, **math**, **code**, or **mixed** — using simple heuristics (function words, symbols, fenced code, etc.). Hints are stored in **`meta.json`** under `lecture_analysis` (`detected_language`, `content_profile`, `has_formulas`, `has_code`, `analysis_updated_at`, plus short notes).
5. The app calls OpenAI **five times** (one per artifact) with prompts that **follow the analysis**: output language matches the detected lecture language; math- and code-heavy rules ask the model to preserve formulas and fenced code, avoid inventing equations, and not rewrite code arbitrarily. All outputs are written only after each call succeeds; if a call fails, partial files from that run are removed and the status is set to **`generation_failed`** (analysis hints are still saved for that run).
6. Rows in the **`artifacts`** table point to each Markdown file. **`meta.json`** is updated with `generation_message` and `generated_artifacts`.

**Re-extract** or **replace source** clears stored `lecture_analysis` until you generate again (stale hints are not kept after the source text changes).

You can **regenerate** when status is **`generation_complete`** or retry after **`generation_failed`** (if extracted text is still present).

### Language- and subject-aware generation

- **Language**: Heuristic detection between **German** and **English** only. Outputs are generated in the detected language (this is **not** a translation feature — the model writes study notes in that language).
- **Subject / style**: **Math** / **code** / **mixed** profiles add system instructions to keep notation, symbols, and code blocks faithful to the source where possible.
- **UI**: The lecture page shows a short **Last generation analysis** block when `lecture_analysis` is present in `meta.json`.

### Text size / truncation

Very long extracted text is truncated to **120,000 characters** before analysis and **before** being sent to the model, with a short note appended. This keeps requests within typical context limits without a full chunking framework. Split the source or re-extract smaller sections if you need full coverage.

### What is preserved well vs not

| Preserved reasonably well | Limitations |
|---------------------------|-------------|
| **Language** matching for DE/EN prose-heavy lectures | Short or mixed-language slides may be misclassified; lists of terms can skew counts |
| **Formulas** when LaTeX-like or symbolic math appears in extracted text | **No** PDF layout, OCR, or LaTeX compilation — only what extraction gives as text |
| **Code** in fenced blocks or obvious syntax patterns | **No** guaranteed recovery of formatting from messy PDFs |
| **Technical vocabulary** when prompts ask not to oversimplify | The model may still paraphrase; heuristics are not semantic understanding |

### What this step does **not** include

No translation mode, OCR, embeddings, chat, graph visualization, or full LaTeX export — only **better prompts** driven by **local heuristics** and **metadata**.

## Search and filters

- **Home**: totals (courses, lectures), **counts by lecture status**, **recent lectures** with status badges, **global lecture search** (`GET /?q=…`) matching **lecture title** or **course name** (SQLite `instr`, substring, case-insensitive), **Needs attention** (status in `extraction_failed`, `generation_failed`, `ready_for_generation`).
- **Course page**: filter lectures by **title substring** (`lec_q`) and/or **status**; filter concepts by **name substring** (`concept_q`) and/or focus **one concept** (`concept` id); **concept chips** jump to the same filters; clear links reset only the relevant filters.

**Limitations**: Substring search only (no stemming, no FTS). Large libraries may feel slower on very wide queries—still a single SQLite file.

## Export and bulk generation

- **Lecture ZIP** (`Download lecture ZIP` on a lecture): includes everything under that lecture’s folder (`source/`, `outputs/`, `meta.json`, `extracted_text.txt`, etc.).
- **Course ZIP** (`Download course ZIP` on a course): zips the whole course directory under `courses/<slug>/` (all lecture subfolders and `course_index/` when present).
- **Bulk generate** (`Generate all ready lectures` on a course): finds lectures with status **`ready_for_generation`**, runs **single-lecture generation** synchronously for each, then shows a **summary** (succeeded / failed / skipped). Skipped = lectures in the course that were **not** ready. Requires **`OPENAI_API_KEY`**. No background jobs—the browser waits until all runs finish.

**Limitations**: Very large courses may produce long downloads or long bulk runs. Bulk generate does not parallelize API calls.

## Current features

- Course and lecture library, upload, extraction (TXT/MD/PDF/DOCX), re-run extraction, replace source, delete lecture
- Lecture statuses including generation lifecycle
- **OpenAI-powered study materials** into `outputs/*.md` (optional)
- **Course concept index** from generated files, SQLite `concepts` / `lecture_concepts`, and `course_index/concept_index.md`
- **Dashboard-style home** and **GET-based search/filters** on home and course pages
- **ZIP export** for one lecture or a whole course; **bulk generate all ready** lectures per course

## Current limitations

- No embeddings, vector search, graph visualization, chat, or cloud sync
- Generation is **synchronous** HTTP (single lecture or bulk sequential); no queue or workers
- Long lectures are **truncated** as described above
- **Language/profile detection** is **heuristic** (German vs English; math/code signals) — not a separate LLM classifier and not perfect on edge cases
- Concept extraction is **heuristic**, not NLP-embedding based
- Search is **substring** match, not ranked relevance
- Costs depend on your OpenAI pricing and usage

## Next recommended build step

Add **SQLite FTS5** for faster title/course search, or **scheduled reminders** / calendar export—before embeddings or chat.
