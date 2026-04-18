"""Compact page-aware help — short answers only; no actions executed here."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request

from app.services import openai_service

_MAX_USER_LEN = 500
_MAX_CONTEXT_JSON = 3500

_SYSTEM = """You are the "Quick help" layer for Study Workflow, a local-first lecture library app (courses, lectures, planner, topic deep dives).

Rules:
- Answer in plain text only. At most 5 short lines (or 2–3 sentences total). No markdown headings, no essays.
- You do NOT rename, delete, or modify anything. Never claim you performed an action.
- If the user wants to rename/delete/change data: tell them exactly what to do in the UI (e.g. open the lecture page → use the delete flow on the course page, or describe where the control is). Say clearly: "I can't change your library from here."
- Use the provided PAGE CONTEXT JSON when relevant (lecture id/title, course, planner hints). If context is missing, answer generally and suggest opening Home or Planner.
- For "what should I study today" / schedule questions: use planner-related context if present; otherwise suggest checking the Planner page and starred lectures on Home.
- Stay calm and practical. No chit-chat, no emojis unless the user used one.

Tone: brief, helpful, like a sidebar note—not a tutor."""


def _trim_context(ctx: dict[str, Any]) -> str:
    raw = json.dumps(ctx, ensure_ascii=False, default=str)
    if len(raw) > _MAX_CONTEXT_JSON:
        return raw[: _MAX_CONTEXT_JSON - 3] + "…"
    return raw


def run_mini_help(*, message: str, context: dict[str, Any] | None) -> tuple[bool, str, str]:
    """
    Returns (ok, reply_text, error_message).
    """
    msg = (message or "").strip()
    if not msg:
        return False, "", "Message is empty."
    if len(msg) > _MAX_USER_LEN:
        return False, "", "Message is too long (max 500 characters)."

    ctx = context if isinstance(context, dict) else {}
    ctx_block = _trim_context(ctx)
    user_block = f"PAGE CONTEXT (JSON):\n{ctx_block}\n\nUSER QUESTION:\n{msg}"

    return openai_service.chat_completion_markdown(
        system_prompt=_SYSTEM,
        user_prompt=user_block,
        max_tokens=380,
    )


def context_for_request(request: Request, page: str, **extra: Any) -> dict[str, Any]:
    """Build JSON-safe context for the mini help bar."""
    out: dict[str, Any] = {"page": page, "path": str(request.url.path)}
    for k, v in extra.items():
        if v is not None:
            out[k] = v
    return out
