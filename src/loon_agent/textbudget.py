"""Approximate token budgeting for small-context local models.

No tokenizer dependency — the classic chars/4 heuristic is plenty for enforcing "never
build a prompt bigger than X". Budgets are enforced by the skill engine on every
template substitution, so no prompt-discipline is ever trusted.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4
_MARKER = " …[truncated]"


def approx_tokens(text: str) -> int:
    """Rough token count (chars/4, rounded up)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int, *, marker: str = _MARKER) -> str:
    """Cut ``text`` to ~``max_tokens``, preferring a whitespace boundary, marking the cut.

    ``max_tokens <= 0`` returns an empty string. Untouched text comes back verbatim.
    """
    if max_tokens <= 0:
        return ""
    if approx_tokens(text) <= max_tokens:
        return text

    limit = max(max_tokens * CHARS_PER_TOKEN - len(marker), 1)
    cut = text[:limit]
    # Back up to the last whitespace so we don't slice mid-word (unless that would
    # discard most of the budget — solid runs get a hard cut).
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary > limit // 2:
        cut = cut[:boundary]
    return cut.rstrip() + marker
