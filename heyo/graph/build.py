"""Assemble the Heyo StateGraph: prepare -> router -> specialized agents -> finalize."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from heyo.config import Settings
from heyo.graph.agents import chat, files
from heyo.graph.context import make_finalize_node, make_prepare_node
from heyo.graph.planner import dispatch, make_planner_node
from heyo.graph.state import AgentState
from heyo.llm.client import LLMClient
from heyo.memory.qdrant import MemoryStore


def build_graph(
    llm: LLMClient,
    settings: Settings,
    memory: MemoryStore | None = None,
    extra_agents: dict | None = None,
):
    """extra_agents: name -> (node_fn, description); lets M4 register web/apps/mcp agents."""
    agents: dict[str, tuple] = {
        "chat": (chat.make_chat_agent(llm), chat.DESCRIPTION),
        "files": (files.make_files_agent(llm, settings.heyo_workspace), files.DESCRIPTION),
    }
    if extra_agents:
        agents.update(extra_agents)

    graph = StateGraph(AgentState)
    graph.add_node("prepare", make_prepare_node(memory))
    graph.add_node("planner", make_planner_node(llm, {n: d for n, (_, d) in agents.items()}))
    for name, (node_fn, _) in agents.items():
        graph.add_node(name, node_fn)
    graph.add_node("finalize", make_finalize_node(memory))

    # prepare -> planner -> [dispatch to step's agent -> dispatch -> ...] -> finalize.
    # Each agent loops back through `dispatch`, which reads plan[cursor] to pick the
    # next agent (or finalize when the plan is exhausted), so a single request can
    # flow across several agents in order.
    next_step = dispatch(agents)
    targets = [*agents, "finalize"]
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "planner")
    graph.add_conditional_edges("planner", next_step, targets)
    for name in agents:
        graph.add_conditional_edges(name, next_step, targets)
    graph.add_edge("finalize", END)
    return graph.compile()
