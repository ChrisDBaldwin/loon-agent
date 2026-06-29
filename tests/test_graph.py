"""Tests for the hand-rolled ReAct loop and session keying — no live backend needed."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from loon_agent.graph import LoonAgent, build_graph
from loon_agent.session import SessionSource, build_session_key
from loon_agent.tools import DEFAULT_TOOLS


class FakeToolModel(BaseChatModel):
    """Replays a scripted list of AI messages — first a tool call, then a final answer."""

    responses: list[AIMessage]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - mirror BaseChatModel signature
        return self

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,
    ) -> ChatResult:
        message = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _scripted_model() -> FakeToolModel:
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "call-1"}],
    )
    final = AIMessage(content="The answer is 4.")
    return FakeToolModel(responses=[tool_call, final])


def test_react_loop_runs_agent_tool_agent_then_ends() -> None:
    graph = build_graph(_scripted_model(), DEFAULT_TOOLS, checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [("user", "what is 2+2?")]},
        {"configurable": {"thread_id": "t1"}},
    )
    messages = result["messages"]

    # The tool actually executed and returned "4".
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "4"

    # The loop ended on a tool-call-free assistant message.
    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls
    assert messages[-1].content == "The answer is 4."


def test_loon_agent_invoke_returns_final_text() -> None:
    agent = LoonAgent(_scripted_model(), DEFAULT_TOOLS, checkpointer=MemorySaver())
    reply = agent.invoke("what is 2+2?", session_key="cli:test")
    assert reply == "The answer is 4."


def test_no_tool_call_routes_straight_to_end() -> None:
    model = FakeToolModel(responses=[AIMessage(content="hello, no tools needed")])
    graph = build_graph(model, DEFAULT_TOOLS, checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [("user", "hi")]},
        {"configurable": {"thread_id": "t2"}},
    )
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])
    assert result["messages"][-1].content == "hello, no tools needed"


def test_session_key_is_stable_and_distinct() -> None:
    a = SessionSource(platform="cli", chat_id="local", user_id="chris")
    a_again = SessionSource(platform="cli", chat_id="local", user_id="chris")
    b = SessionSource(platform="telegram", chat_id="42", user_id="chris")

    assert build_session_key(a) == build_session_key(a_again)
    assert build_session_key(a) != build_session_key(b)
    assert build_session_key(a).startswith("cli:")
