"""Supervisor node: routes each request to a specialized agent via structured output."""

from __future__ import annotations

from typing import Any

from heyo.graph.state import AgentState, trace
from heyo.llm.client import LLMClient, LLMError

ROUTER_PROMPT = """\
You are the routing supervisor of Heyo, a local agentic assistant.
Pick the single best agent for the user's request:

{agent_descriptions}

Consider the conversation history. Default to "chat" when no specialized agent fits.
"""


def route_schema(agent_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": agent_names},
            "rationale": {"type": "string", "description": "one short sentence"},
        },
        "required": ["route", "rationale"],
    }


def make_router_node(llm: LLMClient, agents: dict[str, str]):
    """agents: name -> one-line description (drives both prompt and schema)."""
    descriptions = "\n".join(f'- "{name}": {desc}' for name, desc in agents.items())
    system = ROUTER_PROMPT.format(agent_descriptions=descriptions)
    schema = route_schema(list(agents))

    async def router_node(state: AgentState, *, writer=None) -> AgentState:
        trace(writer, "router", "start")
        messages = [{"role": "system", "content": system}, *state["messages"][-10:]]
        try:
            result = await llm.chat_structured("router", messages, schema)
            route = result.get("route", "chat")
            rationale = result.get("rationale", "")
        except (LLMError, KeyError, ValueError) as exc:
            route, rationale = "chat", f"router failed ({exc}); defaulting to chat"
        if route not in agents:
            route, rationale = "chat", f"unknown route {route!r}; defaulting to chat"
        trace(writer, "router", "done", route=route, rationale=rationale)
        return {"route": route, "rationale": rationale}

    return router_node
