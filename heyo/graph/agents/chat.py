"""General conversation agent: no tools, streams tokens directly."""

from __future__ import annotations

from heyo.graph.agents.base import relevant_skills
from heyo.graph.state import AgentState, emit, trace
from heyo.llm.client import LLMClient

CHAT_PROMPT = """\
You are Heyo, a helpful on-premise voice-and-chat assistant. Be concise and direct;
answers may be read aloud, so avoid markdown tables and long lists unless asked.
"""

DESCRIPTION = "general conversation, questions, reasoning, anything not requiring tools"


def make_chat_agent(llm: LLMClient):
    async def chat_node(state: AgentState, *, writer=None) -> AgentState:
        trace(writer, "chat", "start")
        system = CHAT_PROMPT
        skill_context = relevant_skills(state, "chat")
        if skill_context:
            system += "\n\n# Taught skills\n" + skill_context
        if state.get("memory_context"):
            system += "\n\n# Relevant memories\n" + state["memory_context"]
        messages = [{"role": "system", "content": system}, *state["messages"][-10:]]

        chunks: list[str] = []
        thinking = False
        async for tok in llm.stream("general", messages):
            # Reasoning models emit a <think> block first: stream it live as
            # "thinking" events (shown dimmed in the UI) but keep it out of the
            # final response (and out of TTS).
            if "<think>" in tok:
                thinking = True
                continue
            if "</think>" in tok:
                thinking = False
                continue
            if thinking:
                emit(writer, "thinking", text=tok)
                continue
            chunks.append(tok)
            emit(writer, "token", text=tok)
        content = "".join(chunks).strip()
        trace(writer, "chat", "done")
        return {
            "response": content,
            "messages": [{"role": "assistant", "content": content}],
        }

    return chat_node
