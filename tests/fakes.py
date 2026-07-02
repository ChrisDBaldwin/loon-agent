"""Shared test doubles."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChat(BaseChatModel):
    """Replays scripted string replies and records every prompt it was sent."""

    replies: list[str]
    calls: list[list[BaseMessage]] = []
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - mirror BaseChatModel signature
        return self

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,
    ) -> ChatResult:
        self.calls.append(list(messages))
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
