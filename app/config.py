"""Application configuration loaded from environment with safe defaults."""

from pathlib import Path

from dotenv import load_dotenv
import os
from typing import Literal

# Load .env from project root (parent of app/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _path_from_env(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    p = Path(raw)
    if not p.is_absolute():
        p = (_ROOT / p).resolve()
    return p


APP_ROOT: Path = _ROOT
DATA_DIR: Path = _path_from_env("APP_DATA_DIR", "./data")
COURSES_DIR: Path = _path_from_env("COURSES_STORAGE_DIR", "./courses")
DATABASE_PATH: Path = _path_from_env("DATABASE_PATH", "./data/app.db")


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COURSES_DIR.mkdir(parents=True, exist_ok=True)


# Optional — study material generation (see openai_service)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY", "").strip() or None
OPENAI_MODEL: str = (os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini")

# Study material generation: "legacy" = current prompt set; "strict_v2" = stricter, structure-anchored prompts + tighter classification.
# Default stays legacy so existing deployments behave unchanged until you opt in (e.g. GENERATION_MODE=strict_v2 in .env).
_GM = (os.getenv("GENERATION_MODE", "legacy").strip().lower() or "legacy")
GENERATION_MODE: Literal["legacy", "strict_v2"] = "strict_v2" if _GM == "strict_v2" else "legacy"
