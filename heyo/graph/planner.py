"""Planner node: decompose a request into an ordered list of agent steps.

Replaces single-shot routing. Output is a plan — `[{agent, task, effort}, ...]` —
so a request that spans several agents ("search X, save it to a file, open Y")
runs each step in turn instead of being forced onto one agent. `effort` per step
drives how much the executing agent thinks (none/brief/deep), so trivial steps
answer instantly and hard ones reason fully.
"""

from __future__ import annotations

import json
import re
from typing import Any

from heyo.graph.state import AgentState, trace
from heyo.llm.client import LLMClient, LLMError

PLANNER_PROMPT = """\
You are the planner of Heyo, a local agentic assistant. Break the user's request
into the FEWEST ordered steps that get it done, each handled by exactly one agent.

Agents:
{agent_descriptions}

For every step give:
- agent: which agent handles it
- task: a self-contained imperative instruction (a later step may rely on an
  earlier step's result — say so in the task, e.g. "save the version you found")
- effort: how hard the agent should think —
    "none"  trivial/mechanical: greetings, opening an app, a single file write
    "brief" a little reasoning or a tool or two
    "deep"  multi-hop research, ambiguity, or real reasoning

Rules:
- MOST requests are ONE step. Only split when the request truly needs different
  agents or clearly separate actions. Never invent steps the user didn't ask for.
- BUT when the request chains distinct actions ("search ... then save ...",
  "do X and then Y", "find Z and open it"), you MUST emit one step per action,
  each on the agent that can actually do it. A "files" task needs the files
  agent; "chat" has no tools and cannot search, save files, or open apps.
- Order steps so each can use the results of the ones before it.
- If the user expects a single combined answer from several results, end with a
  "chat" step that presents them.

Examples:
- "what's 12 percent of 80?" -> [{{"agent":"chat","task":"answer what 12% of 80 is","effort":"none"}}]
- "open notepad" -> [{{"agent":"apps","task":"open notepad","effort":"none"}}]
- "search the web for the latest python version and save it to ver.txt" ->
  [{{"agent":"web","task":"find the latest Python version","effort":"brief"}},
   {{"agent":"files","task":"save the Python version you found to ver.txt","effort":"none"}}]
- "compare the populations of France and Japan" ->
  [{{"agent":"web","task":"find the populations of France and Japan","effort":"brief"}},
   {{"agent":"chat","task":"compare the two populations you found","effort":"brief"}}]
"""


_JSON_NUDGE = (
    'Output ONLY this JSON, nothing else: '
    '{"steps":[{"agent":"<name>","task":"<imperative>","effort":"none|brief|deep"}]}'
)


def _extract_json(content: str) -> dict[str, Any]:
    """Pull the JSON object out of the model's reply (tolerating a <think> block
    or ``` fences). Dumping a full JSON schema into the prompt — what
    chat_structured does — reliably confuses qwen3:4b into empty plans, so the
    planner prompts in plain language and parses the result itself."""
    text = content.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:].strip() if text[:4] == "json" else text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in: {content[:160]!r}")
    return json.loads(text[start : end + 1])


# Trivial single-action shortcuts: skip the planner LLM only when the request is
# clearly ONE action (no conjunction hinting at a second task).
FAST_ROUTES: list[tuple[str, str, str]] = [
    (r"\b(open|launch|start)\b.{0,40}\b(app|application|calculator|notepad|browser|program)\b",
     "apps", "none"),
    (r"\b(search|google|look\s*up|browse)\b|https?://|\bwebsite\b", "web", "brief"),
]
_MULTI = re.compile(r"\b(and|then|also|after|afterwards|next|plus)\b|[;,]")

MAX_STEPS = 5


def make_planner_node(llm: LLMClient, agents: dict[str, str]):
    """agents: name -> one-line description (drives both prompt and schema)."""
    descriptions = "\n".join(f'- "{name}": {desc}' for name, desc in agents.items())
    system = PLANNER_PROMPT.format(agent_descriptions=descriptions)

    def _clean(steps: list[dict], user_msg: str) -> list[dict]:
        out: list[dict] = []
        for s in steps[:MAX_STEPS]:
            agent = s.get("agent")
            task = (s.get("task") or "").strip()
            if agent not in agents:
                agent = "chat"
            effort = s.get("effort") if s.get("effort") in ("none", "brief", "deep") else "brief"
            out.append({"agent": agent, "task": task or user_msg, "effort": effort})
        return out or [{"agent": "chat", "task": user_msg, "effort": "brief"}]

    async def planner_node(state: AgentState, *, writer=None) -> AgentState:
        trace(writer, "planner", "start")
        user_msg = state["messages"][-1]["content"]
        low = user_msg.lower()

        plan: list[dict] | None = None
        if not _MULTI.search(low):  # only shortcut clearly single-action requests
            for pattern, agent, effort in FAST_ROUTES:
                if agent in agents and re.search(pattern, low):
                    plan = [{"agent": agent, "task": user_msg, "effort": effort}]
                    break

        if plan is None:
            messages = [
                {"role": "system", "content": system},
                *state["messages"][-4:],
                {"role": "user", "content": _JSON_NUDGE},
            ]
            try:
                msg = await llm.chat("router", messages, think=False)
                plan = _clean(_extract_json(msg.get("content") or "").get("steps") or [], user_msg)
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                trace(writer, "planner", "done", error=f"planner failed ({exc}); single chat step")
                plan = [{"agent": "chat", "task": user_msg, "effort": "brief"}]

        summary = " → ".join(s["agent"] for s in plan)
        trace(writer, "planner", "done", plan=plan, route=plan[0]["agent"], rationale=summary)
        return {
            "plan": plan,
            "cursor": 0,
            "step_results": [],
            "route": plan[0]["agent"],
            "rationale": summary,
        }

    return planner_node


def dispatch(agents: dict) -> Any:
    """Conditional-edge fn: next step's agent, or 'finalize' when the plan is done."""

    def _next(state: AgentState) -> str:
        plan = state.get("plan") or []
        cur = state.get("cursor", 0)
        if 0 <= cur < len(plan) and plan[cur]["agent"] in agents:
            return plan[cur]["agent"]
        return "finalize"

    return _next
