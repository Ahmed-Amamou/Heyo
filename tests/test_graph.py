from __future__ import annotations

import json

from heyo.config import Settings
from heyo.graph.build import build_graph

from tests.conftest import FakeLLM


def make_settings(workspace) -> Settings:
    return Settings(heyo_workspace=workspace)


async def run_graph(graph, message: str):
    events, final = [], {}
    async for mode, payload in graph.astream(
        {"session_id": "t", "messages": [{"role": "user", "content": message}]},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            events.append(payload)
        else:
            final = payload
    return events, final


async def test_chat_route_streams_tokens(workspace):
    llm = FakeLLM(route_to="chat", stream_text="hi there")
    graph = build_graph(llm, make_settings(workspace))
    events, final = await run_graph(graph, "hello")

    routes = [e for e in events if e.get("node") == "router" and e.get("status") == "done"]
    assert routes and routes[0]["route"] == "chat"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "hi there" in tokens
    assert final["response"] == "hi there"


async def test_files_route_executes_tool(workspace):
    write_call = {
        "content": None,
        "tool_calls": [{
            "id": "1",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"path": "notes.txt", "content": "hello"}),
            },
        }],
    }
    llm = FakeLLM(route_to="files", replies=[write_call, {"content": "created notes.txt"}])
    graph = build_graph(llm, make_settings(workspace))
    events, final = await run_graph(graph, "create notes.txt containing hello")

    assert (workspace / "notes.txt").read_text() == "hello"
    assert final["response"] == "created notes.txt"
    tool_events = [e for e in events if e.get("status") == "tool"]
    assert tool_events and tool_events[0]["tool"] == "write_file"


async def test_files_tool_sandbox_blocks_escape(workspace):
    escape_call = {
        "content": None,
        "tool_calls": [{
            "id": "1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "../../etc/passwd"}),
            },
        }],
    }
    llm = FakeLLM(route_to="files", replies=[escape_call, {"content": "blocked"}])
    graph = build_graph(llm, make_settings(workspace))
    events, _ = await run_graph(graph, "read /etc/passwd")

    results = [e for e in events if e.get("status") == "tool_result"]
    assert results and "error" in results[0]["result"]


async def test_router_bad_output_defaults_to_chat(workspace):
    llm = FakeLLM(route_to="nonexistent_agent")
    graph = build_graph(llm, make_settings(workspace))
    events, final = await run_graph(graph, "hello")
    routes = [e for e in events if e.get("node") == "router" and e.get("status") == "done"]
    assert routes[0]["route"] == "chat"
    assert final["response"]
