"""General conversation agent: no tools, streams tokens directly."""

from __future__ import annotations

from heyo.graph.agents.base import (
    build_step_context,
    current_step,
    effort_settings,
    finish_step,
    is_final_step,
)
from heyo.graph.state import AgentState, emit, trace
from heyo.llm.client import LLMClient

CHAT_PROMPT = """\
You are Heyo, a helpful on-premise voice-and-chat assistant. Be concise and direct;
answers may be read aloud, so avoid markdown tables and long lists unless asked.
"""

DESCRIPTION = "general conversation, questions, reasoning, anything not requiring tools"


def make_chat_agent(llm: LLMClient):
    async def chat_node(state: AgentState, *, writer=None) -> AgentState:
        step = current_step(state)
        final = is_final_step(state)
        think, hint = effort_settings(step, has_tools=False)  # chat: always think (clean prose)
        system, convo, task = build_step_context(state, CHAT_PROMPT, "chat", hint)
        trace(writer, "chat", "start", task=task, effort=(step or {}).get("effort"))

        # Reasoning streams live as "thinking" events (dimmed in the UI); answer
        # tokens stream only on the final step (earlier steps feed forward).
        content = ""
        streamed = False
        async for kind, data in llm.stream_message("general", convo, think=think):
            if kind == "thinking":
                emit(writer, "thinking", text=data)
            elif kind == "token":
                if final:
                    streamed = True
                    emit(writer, "token", text=data)
            elif kind == "message":
                content = data["content"]
        if final and not streamed:
            emit(writer, "token", text=content)
        return finish_step(state, "chat", content, task, final, writer)

    return chat_node
