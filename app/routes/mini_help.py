"""JSON API for the compact Quick help bar (no server-side actions)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services import mini_help_service, openai_service

router = APIRouter(prefix="/mini-help", tags=["mini-help"])


class MiniHelpChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    context: dict[str, Any] = Field(default_factory=dict)


@router.post("/chat")
def post_mini_help_chat(body: MiniHelpChatIn) -> dict[str, Any]:
    if not openai_service.is_openai_configured():
        return {
            "ok": False,
            "reply": "",
            "error": "OpenAI is not configured. Add OPENAI_API_KEY to your .env file to use Quick help.",
        }

    ok, reply, err = mini_help_service.run_mini_help(
        message=body.message,
        context=body.context,
    )
    if ok:
        return {"ok": True, "reply": reply, "error": ""}
    return {"ok": False, "reply": "", "error": err or "Request failed."}
