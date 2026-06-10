"""General conversation agent: no tools, streams tokens directly."""

from __future__ import annotations

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
        if state.get("skill_context"):
            system += "\n\n# Taught skills\n" + state["skill_context"]
        if state.get("memory_context"):
            system += "\n\n# Relevant memories\n" + state["memory_context"]
        messages = [{"role": "system", "content": system}, *state["messages"][-20:]]

        chunks: list[str] = []
        thinking = False
        async for tok in llm.stream("general", messages):
            # Strip <think> blocks from reasoning models before they reach the user.
            if "<think>" in tok:
                thinking = True
                continue
            if "</think>" in tok:
                thinking = False
                continue
            if thinking:
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
