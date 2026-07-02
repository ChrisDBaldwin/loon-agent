"""The hand-rolled ReAct loop.

This is the learning centerpiece: rather than calling ``create_agent``, we assemble the
classic reason -> act -> observe loop out of LangGraph primitives directly:

    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

``agent`` calls the tool-bound model; ``tools_condition`` routes to the ``ToolNode`` when
the model emitted tool calls, otherwise to ``END``; the tool results feed back into
``agent``. A checkpointer makes each ``thread_id`` a durable conversation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .state import AgentState

if TYPE_CHECKING:
    from .memory.provider import MemoryProvider

SYSTEM_PROMPT = (
    "You are loon, a helpful homelab assistant running on the user's own hardware. "
    "Be concise and direct. Use the available tools when they let you give a grounded, "
    "correct answer rather than guessing."
)


def _build_messages(
    state: AgentState,
    config: RunnableConfig,
    memory: MemoryProvider | None,
    persona: str | None = None,
) -> list[BaseMessage]:
    """Assemble the message list for the model: system + persona + memory + history."""
    blocks = [SYSTEM_PROMPT, persona or ""]
    if memory is not None:
        session_id = (config.get("configurable") or {}).get("thread_id", "default")
        if static_block := memory.system_prompt_block():
            blocks.append(static_block)
        query = _latest_user_text(state["messages"])
        if recall := memory.prefetch(query, session_id):
            blocks.append("Relevant context from memory:\n" + recall)
    system = SystemMessage("\n\n".join(b for b in blocks if b))
    return [system, *state["messages"]]


def _text(message: BaseMessage) -> str:
    """Best-effort plain text of a message across langchain content-block variants.

    ``.text`` is a property in langchain-core 1.x but was a method earlier; handle both.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        return text()
    return str(message.content)


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text(message)
    return ""


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    memory: MemoryProvider | None = None,
    persona: str | None = None,
):
    """Compile the ReAct StateGraph for the given model, tools and (optional) memory.

    ``persona`` is an optional system-prompt block — e.g. a donned masque's lens.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        messages = _build_messages(state, config, memory, persona)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


class LoonAgent:
    """Convenience wrapper: a compiled graph plus optional memory write-back per turn."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        memory: MemoryProvider | None = None,
        persona: str | None = None,
    ) -> None:
        self.memory = memory
        self.graph = build_graph(
            llm, tools, checkpointer=checkpointer, memory=memory, persona=persona
        )

    def invoke(self, text: str, session_key: str) -> str:
        """Run one full turn and return the assistant's final text reply."""
        config = {"configurable": {"thread_id": session_key}}
        result = self.graph.invoke({"messages": [HumanMessage(text)]}, config)
        reply = _final_text(result["messages"])
        self._write_back(text, reply, session_key)
        return reply

    def stream(self, text: str, session_key: str) -> Iterator[BaseMessage]:
        """Run one turn, yielding each new message (AI/tool) as nodes complete.

        Lets a UI surface tool calls and results in real time. Memory write-back runs
        once the turn finishes.
        """
        config = {"configurable": {"thread_id": session_key}}
        reply = ""
        # stream_mode="updates" yields one dict per super-step: {node_name: {"messages": [...]}}.
        for update in self.graph.stream(
            {"messages": [HumanMessage(text)]}, config, stream_mode="updates"
        ):
            for value in update.values():
                for message in value.get("messages", []):
                    yield message
                    if isinstance(message, AIMessage) and not message.tool_calls:
                        reply = _text(message)
        self._write_back(text, reply, session_key)

    def _write_back(self, user_text: str, reply: str, session_key: str) -> None:
        if self.memory is not None:
            self.memory.sync_turn(user_text, reply, session_id=session_key)


def _final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _text(message)
    return ""
