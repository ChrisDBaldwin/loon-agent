"""LLM construction: one ``ChatOpenAI`` pointed at any homelab backend.

We use ``ChatOpenAI`` (not ``langchain_community.VLLMOpenAI``) deliberately — only
``ChatOpenAI`` supports ``bind_tools``, which the agent loop needs. Local servers ignore
the API key but the SDK still requires a non-empty value.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Settings, get_settings

# Local OpenAI-compatible servers ignore auth; the SDK just needs a non-empty key.
_LOCAL_API_KEY = "not-needed"


def make_llm(
    backend: str | None = None,
    *,
    settings: Settings | None = None,
    **kwargs: object,
) -> ChatOpenAI:
    """Build a chat model for the named backend (default: configured ``LOON_BACKEND``).

    Extra keyword args are forwarded to ``ChatOpenAI`` (e.g. ``temperature``,
    ``streaming``, ``extra_body`` for vLLM-specific knobs).
    """
    settings = settings or get_settings()
    target = settings.resolve_backend(backend)
    params: dict[str, object] = {
        "model": target.model,
        "base_url": target.base_url,
        "api_key": _LOCAL_API_KEY,
        "temperature": settings.temperature,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)
