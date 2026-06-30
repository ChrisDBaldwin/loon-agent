"""LLM construction: one ``ChatOpenAI`` pointed at any homelab backend.

We use ``ChatOpenAI`` (not ``langchain_community.VLLMOpenAI``) deliberately — only
``ChatOpenAI`` supports ``bind_tools``, which the agent loop needs. Most local servers
ignore auth (the SDK still requires a non-empty value), but a backend can carry a real
token via ``LOON_<NAME>_API_KEY`` — e.g. LM Studio with API-token auth enabled.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Settings, get_settings


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
        "api_key": target.api_key,
        "temperature": settings.temperature,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)
