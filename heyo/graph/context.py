"""prepare/finalize nodes: retrieve memories + skills before routing, write memory after.

All Qdrant access degrades gracefully — if the vector store or embedder is down,
the graph still answers, it just runs without memory/skill context.
"""

from __future__ import annotations

from heyo.graph.state import AgentState, trace
from heyo.memory.qdrant import MemoryStore


def make_prepare_node(memory: MemoryStore | None):
    async def prepare_node(state: AgentState, *, writer=None) -> AgentState:
        if memory is None:
            return {}
        trace(writer, "prepare", "start")
        query = state["messages"][-1]["content"]
        memory_context = ""
        skills: list = []
        try:
            # keep this small: it lands in every agent prompt, and long prompts both
            # slow prefill and push reasoning models into context-shift territory
            memories = await memory.recall(query, limit=3)
            memory_context = "\n".join(f"- {m[:240]}" for m in memories)
            skills = await memory.find_skills(query)
            trace(writer, "prepare", "done",
                  memories=len(memories), skills=[s["name"] for s in skills])
        except Exception as exc:
            trace(writer, "prepare", "done", error=f"memory unavailable: {exc}")
        return {"memory_context": memory_context, "skills": skills}

    return prepare_node


def make_finalize_node(memory: MemoryStore | None):
    async def finalize_node(state: AgentState, *, writer=None) -> AgentState:
        if memory is not None and state.get("response"):
            user_msg = next(
                (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
            )
            exchange = f"User: {user_msg[:300]}\nAssistant: {state['response'][:300]}"
            try:
                await memory.remember(state.get("session_id", ""), exchange)
                trace(writer, "finalize", "done", memorized=True)
                return {}
            except Exception as exc:
                trace(writer, "finalize", "done", error=f"memory write failed: {exc}")
                return {}
        trace(writer, "finalize", "done")
        return {}

    return finalize_node
