"""Tool registry + generic ReAct-style tool-calling agent loop."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from heyo.graph.state import AgentState, emit, trace
from heyo.llm.client import LLMClient
from heyo.skills.loader import format_skills


def relevant_skills(state: AgentState, agent_name: str) -> str:
    """Only the skills taught for this agent — keeps prompts small, which matters:
    prefill is the bottleneck on consumer GPUs (~266 tok/s on a GTX 1660 Ti)."""
    skills = [s for s in state.get("skills", []) if s.get("agent") in (agent_name, "any")]
    return format_skills(skills) if skills else ""


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


def parse_soft_tool_call(content: str, toolkit: ToolKit) -> tuple[str, dict] | None:
    """Detect a tool call written as JSON text instead of the tool_calls channel.

    Small local models frequently do this (e.g. ```json {"name": "write_file",
    "arguments": {...}} ```). Returns (tool_name, arguments) or None.
    """
    text = content.strip()
    if "```" in text:
        for block in text.split("```")[1::2]:
            block = block.removeprefix("json").strip()
            found = parse_soft_tool_call(block, toolkit)
            if found:
                return found
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("tool") or data.get("function")
    args = data.get("arguments") or data.get("parameters") or {}
    if isinstance(name, str) and name in toolkit.tools and isinstance(args, dict):
        return name, args
    return None


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
        skill_context = relevant_skills(state, name)
        if skill_context:
            system += "\n\n# Taught skills (follow these precisely)\n" + skill_context
        if state.get("memory_context"):
            system += "\n\n# Relevant memories\n" + state["memory_context"]
        convo: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *state["messages"][-8:],
        ]

        for _ in range(max_iterations):
            # think=False: hidden reasoning adds seconds per tool round with no benefit here
            msg = await llm.chat(role, convo, tools=toolkit.specs() or None, think=False)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                soft = parse_soft_tool_call(content, toolkit)
                if soft:
                    tool_name, args = soft
                    trace(writer, name, "tool", tool=tool_name, args=json.dumps(args))
                    result = await toolkit.execute(tool_name, args)
                    trace(writer, name, "tool_result", tool=tool_name, result=result[:500])
                    convo.append({"role": "assistant", "content": content})
                    convo.append(
                        {"role": "user", "content": f"Tool result for {tool_name}: {result}\n"
                         "Reply with the final answer for the user (or another tool call)."}
                    )
                    continue
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
