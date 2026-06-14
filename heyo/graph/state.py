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
    plan: list[dict[str, Any]]  # planner's ordered steps: [{agent, task, effort}]
    cursor: int  # index of the step currently executing
    step_results: Annotated[list[dict[str, Any]], _append]  # [{agent, task, result}]
    route: str  # first step's agent (kept for the trace / back-compat)
    rationale: str  # planner's one-line summary, surfaced in the UI trace
    skills: list[dict[str, Any]]  # taught .md skills retrieved for this request
    memory_context: str  # relevant past-conversation memories
    response: str  # final assistant answer (last step's output)


def emit(writer: StreamWriter | None, event_type: str, **data: Any) -> None:
    """Push a custom stream event (trace/token) to whoever is consuming the graph."""
    if writer:
        writer({"type": event_type, "ts": time.time(), **data})


def trace(writer: StreamWriter | None, node: str, status: str, **data: Any) -> None:
    emit(writer, "trace", node=node, status=status, **data)
