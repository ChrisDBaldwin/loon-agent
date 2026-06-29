"""Agent graph state.

We start from LangGraph's ``MessagesState`` (a ``messages`` channel with the
``add_messages`` reducer) and subclass it so extra channels can be added later without
touching the reducer wiring. Keep this lean — every field is serialized into the
checkpoint on each node transition.
"""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation state. Inherits ``messages: Annotated[list, add_messages]``."""
