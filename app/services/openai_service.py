"""Thin OpenAI client wrapper — swap models or providers here later."""

from __future__ import annotations

from typing import Optional, Tuple

from app.config import OPENAI_API_KEY, OPENAI_MODEL

# Lazy client so importing the app does not require the package until first use
_client = None


def get_openai_api_key() -> Optional[str]:
    return OPENAI_API_KEY


def get_openai_model() -> str:
    # Override via .env OPENAI_MODEL (e.g. gpt-5-mini when your account supports it)
    return OPENAI_MODEL


def is_openai_configured() -> bool:
    return get_openai_api_key() is not None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = get_openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=key, timeout=120.0)
    return _client


def reset_client_for_tests() -> None:
    global _client
    _client = None


def chat_completion_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> Tuple[bool, str, str]:
    """
    Single chat completion; expects Markdown in the reply.
    Returns (ok, markdown_text, error_message).
    """
    if not is_openai_configured():
        return False, "", "OpenAI is not configured. Add OPENAI_API_KEY to your .env file."

    try:
        client = _get_client()
        model = get_openai_model()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        text = (choice.content or "").strip()
        if not text:
            return False, "", "The model returned empty text."
        return True, text, ""
    except Exception as e:  # noqa: BLE001
        return False, "", f"OpenAI request failed: {e}"
