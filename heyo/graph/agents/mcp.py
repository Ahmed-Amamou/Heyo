"""MCP agent: exposes tools from the user's MCP servers (mcp.json) to the graph."""

from __future__ import annotations

from heyo.graph.agents.base import make_tool_agent
from heyo.llm.client import LLMClient
from heyo.mcp.manager import MCPManager

MCP_PROMPT = """\
You are Heyo's external-tools agent. Your tools come from the user's own MCP servers.
Pick the right tool for the request, call it with correct arguments, and report the
result concisely. If no tool fits, say which tools you do have.
"""


def description(manager: MCPManager) -> str:
    names = ", ".join(manager.tool_names()[:15]) or "none configured yet"
    return f"use the user's external MCP tools ({names})"


def make_mcp_agent(llm: LLMClient, manager: MCPManager):
    return make_tool_agent("mcp", llm, "general", MCP_PROMPT, manager.toolkit())
