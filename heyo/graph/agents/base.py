"""Tool registry + generic ReAct-style tool-calling agent loop."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from heyo.graph.state import AgentState, emit, trace
from heyo.llm.client import LLMClient


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for arguments
    fn: Callable[..., Any]

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolKit:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, name: str, description: str, parameters: dict[str, Any]):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[name] = Tool(name, description, parameters, fn)
            return fn

        return deco

    def specs(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    async def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            result = tool.fn(**args)
            if inspect.isawaitable(result):
                result = await result
            else:
                # Keep the event loop responsive around blocking tools.
                await asyncio.sleep(0)
            return str(result)
        except Exception as exc:  # tool errors go back to the model, not up the stack
            return f"error: {exc}"


def make_tool_agent(
    name: str,
    llm: LLMClient,
    role: str,
    system_prompt: str,
    toolkit: ToolKit,
    max_iterations: int = 8,
):
    """Build a graph node that runs an LLM tool-calling loop and streams its final answer."""

    async def agent_node(state: AgentState, *, writer=None) -> AgentState:
        trace(writer, name, "start")
        system = system_prompt
        if state.get("skill_context"):
            system += "\n\n# Taught skills (follow these precisely)\n" + state["skill_context"]
        if state.get("memory_context"):
            system += "\n\n# Relevant memories\n" + state["memory_context"]
        convo: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *state["messages"][-20:],
        ]

        for _ in range(max_iterations):
            msg = await llm.chat(role, convo, tools=toolkit.specs() or None)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                emit(writer, "token", text=content)
                trace(writer, name, "done")
                return {
                    "response": content,
                    "messages": [{"role": "assistant", "content": content}],
                }
            convo.append(msg)
            for call in tool_calls:
                fn = call["function"]
                trace(writer, name, "tool", tool=fn["name"], args=fn.get("arguments", ""))
                result = await toolkit.execute(fn["name"], fn.get("arguments", "{}"))
                trace(writer, name, "tool_result", tool=fn["name"], result=result[:500])
                convo.append(
                    {"role": "tool", "tool_call_id": call.get("id", fn["name"]), "content": result}
                )

        fallback = "I hit the tool-iteration limit before finishing. Here is where I got:\n" + (
            convo[-1].get("content") or ""
        )
        trace(writer, name, "done", note="max_iterations reached")
        return {"response": fallback, "messages": [{"role": "assistant", "content": fallback}]}

    return agent_node
