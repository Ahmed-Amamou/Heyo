"""Shared LangGraph state and stream-event helpers.

Nodes receive a `writer` keyword argument (LangGraph's StreamWriter injection —
required on Python 3.10, where get_stream_writer() doesn't work in async nodes).
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Callable, TypedDict

StreamWriter = Callable[[dict[str, Any]], None]


def _append(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, Any]], _append]  # OpenAI-format chat history
    route: str  # chosen agent: chat|files|web|apps|mcp
    rationale: str  # router's reasoning, surfaced in the UI trace
    skill_context: str  # taught .md skills retrieved for this request (M3)
    memory_context: str  # relevant past-conversation memories (M3)
    response: str  # final assistant answer


def emit(writer: StreamWriter | None, event_type: str, **data: Any) -> None:
    """Push a custom stream event (trace/token) to whoever is consuming the graph."""
    if writer:
        writer({"type": event_type, "ts": time.time(), **data})


def trace(writer: StreamWriter | None, node: str, status: str, **data: Any) -> None:
    emit(writer, "trace", node=node, status=status, **data)
