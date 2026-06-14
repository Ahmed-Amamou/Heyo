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


# --- plan-execute helpers (shared by every agent) --------------------------

def current_step(state: AgentState) -> dict | None:
    plan = state.get("plan") or []
    cur = state.get("cursor", 0)
    return plan[cur] if 0 <= cur < len(plan) else None


def is_final_step(state: AgentState) -> bool:
    """True for the last step (or when there's no plan) — only the final step's
    answer streams to the user as the response; earlier steps feed forward."""
    plan = state.get("plan") or []
    return state.get("cursor", 0) >= len(plan) - 1


def effort_settings(step: dict | None, has_tools: bool) -> tuple[bool, str]:
    """Map a step's effort to (think, system-hint).

    think=False genuinely turns qwen3 reasoning off (fast) but only stays clean
    for tool-driven steps — a mechanical 'none' step just calls its tool. Prose
    agents (chat) always keep thinking on, or qwen3 rambles its reasoning into the
    answer. 'brief' nudges shorter reasoning; 'deep' reasons freely."""
    effort = (step or {}).get("effort", "brief")
    if effort == "none":
        hint = "This is a simple request — answer directly, with minimal deliberation."
    elif effort == "brief":
        hint = "Think briefly — a sentence or two of reasoning at most, then act."
    else:
        hint = ""
    think = (effort != "none") if has_tools else True
    return think, hint


def build_step_context(state: AgentState, base_system: str, agent_name: str,
                       hint: str) -> tuple[str, list[dict], str]:
    """Assemble the system prompt (skills + memory + prior results + effort hint)
    and the conversation for the current step. Returns (system, convo, task)."""
    step = current_step(state)
    task = step["task"] if step else state["messages"][-1]["content"]
    system = base_system
    if hint:
        system += "\n\n" + hint
    skill_context = relevant_skills(state, agent_name)
    if skill_context:
        system += "\n\n# Taught skills (follow precisely)\n" + skill_context
    if state.get("memory_context"):
        system += "\n\n# Relevant memories\n" + state["memory_context"]
    prior = state.get("step_results") or []
    if prior:
        system += "\n\n# Results from earlier steps of this request\n" + "\n".join(
            f"- {r['agent']}: {r['task']} -> {r['result'][:300]}" for r in prior)

    convo: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if len(state.get("plan") or []) > 1:  # multi-step: focus on this sub-task only
        convo += state["messages"][-4:]
        convo.append({"role": "user", "content":
                      f"This is ONE step of a larger plan. Do ONLY this step and report "
                      f"only what you actually did — do not claim to do other steps "
                      f"(later steps handle the rest). Step: {task}"})
    else:  # single step: behave like a normal one-shot agent over history
        convo += state["messages"][-8:]
    return system, convo, task


def finish_step(state: AgentState, agent_name: str, content: str, task: str,
                final: bool, writer) -> AgentState:
    """Record a step's result, advance the cursor, and (only on the final step)
    commit the answer to history."""
    updates: AgentState = {
        "response": content,
        "step_results": [{"agent": agent_name, "task": task, "result": content}],
        "cursor": state.get("cursor", 0) + 1,
    }
    if final:
        updates["messages"] = [{"role": "assistant", "content": content}]
        trace(writer, agent_name, "done")
    else:
        trace(writer, agent_name, "result", result=content[:500])
    return updates


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


def _json_objects(text: str):
    """Yield every balanced top-level {...} block in text."""
    depth, start = 0, -1
    in_str = escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]


def _unwrap_args(args: dict) -> dict:
    """Models sometimes emit schema-shaped values: {"url": {"type":"string","value":"x"}}."""
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in args.items()
    }


def parse_soft_tool_call(content: str, toolkit: ToolKit) -> tuple[str, dict] | None:
    """Detect a tool call written as JSON text instead of the tool_calls channel.

    Small local models frequently do this (e.g. ```json {"name": "write_file",
    "arguments": {...}} ```), sometimes several per message — the first valid one
    wins and the loop handles the rest on later rounds.
    Returns (tool_name, arguments) or None.
    """
    for block in _json_objects(content):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name") or data.get("tool") or data.get("function")
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(name, str) and name in toolkit.tools and isinstance(args, dict):
            return name, _unwrap_args(args)
    return None


def make_tool_agent(
    name: str,
    llm: LLMClient,
    role: str,
    system_prompt: str,
    toolkit: ToolKit,
    max_iterations: int = 8,
    force_first_tool: str | None = None,
):
    """Build a graph node that runs an LLM tool-calling loop and streams its final answer.

    force_first_tool: if set, the agent may not answer before at least one tool ran —
    that tool is invoked with {"query": <user message>} first. Small models love to
    answer 'web' questions from stale memory; this guarantees grounding.
    """

    async def agent_node(state: AgentState, *, writer=None) -> AgentState:
        step = current_step(state)
        final = is_final_step(state)
        think, hint = effort_settings(step, bool(toolkit.tools))
        system, convo, task = build_step_context(state, system_prompt, name, hint)
        trace(writer, name, "start", task=task, effort=(step or {}).get("effort"))

        ran_a_tool = False
        for _ in range(max_iterations):
            # ReAct round, streamed live: reasoning -> "thinking" events; answer
            # tokens -> "token" events (only on the final step — earlier steps feed
            # their result forward instead of speaking it); tool calls from the
            # same stream. think follows the step's effort.
            msg: dict[str, Any] = {}
            streamed_answer = False
            async for kind, data in llm.stream_message(
                role, convo, tools=toolkit.specs() or None, think=think
            ):
                if kind == "thinking":
                    emit(writer, "thinking", text=data)
                elif kind == "token":
                    if final:
                        streamed_answer = True
                        emit(writer, "token", text=data)
                elif kind == "message":
                    msg = data
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                if not ran_a_tool and force_first_tool and force_first_tool in toolkit.tools:
                    trace(writer, name, "tool", tool=force_first_tool,
                          args=json.dumps({"query": task}), forced=True)
                    result = await toolkit.execute(force_first_tool, {"query": task})
                    trace(writer, name, "tool_result", tool=force_first_tool, result=result[:500])
                    ran_a_tool = True
                    convo.append(
                        {"role": "user", "content":
                         f"Do not answer from memory. Result of {force_first_tool}: {result}\n"
                         "Use this (and further tool calls if needed) to answer."}
                    )
                    continue
                soft = parse_soft_tool_call(content, toolkit)
                if soft:
                    tool_name, args = soft
                    trace(writer, name, "tool", tool=tool_name, args=json.dumps(args))
                    result = await toolkit.execute(tool_name, args)
                    trace(writer, name, "tool_result", tool=tool_name, result=result[:500])
                    ran_a_tool = True
                    convo.append({"role": "assistant", "content": content})
                    convo.append(
                        {"role": "user", "content": f"Tool result for {tool_name}: {result}\n"
                         "Reply with the final answer for the user (or another tool call)."}
                    )
                    continue
                if final and not streamed_answer:
                    emit(writer, "token", text=content)
                return finish_step(state, name, content, task, final, writer)
            convo.append(msg)
            for call in tool_calls:
                fn = call["function"]
                trace(writer, name, "tool", tool=fn["name"], args=fn.get("arguments", ""))
                result = await toolkit.execute(fn["name"], fn.get("arguments", "{}"))
                ran_a_tool = True
                trace(writer, name, "tool_result", tool=fn["name"], result=result[:500])
                convo.append(
                    {"role": "tool", "tool_call_id": call.get("id", fn["name"]), "content": result}
                )

        fallback = "I hit the tool-iteration limit before finishing. Here is where I got:\n" + (
            convo[-1].get("content") or ""
        )
        if final:
            emit(writer, "token", text=fallback)
        return finish_step(state, name, fallback, task, final, writer)

    return agent_node
