"""FastAPI entry point for study_workflow_app.

Single-user local app: no login, sessions, or multi-tenant routing (see README).
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import APP_ROOT, ensure_directories
from app.routes import courses, home, lectures, mini_help, planner, uni_tasks, upload
from app.services.database_service import initialize_database

app = FastAPI(title="Study Workflow App", description="Local lecture library and study workflow foundation.")

@app.on_event("startup")
def _startup() -> None:
    ensure_directories()
    initialize_database()


app.mount(
    "/static",
    StaticFiles(directory=str(APP_ROOT / "app" / "static")),
    name="static",
)

app.include_router(home.router)
app.include_router(mini_help.router)
app.include_router(upload.router)
app.include_router(courses.router)
app.include_router(lectures.router)
app.include_router(planner.router)
app.include_router(uni_tasks.router)
