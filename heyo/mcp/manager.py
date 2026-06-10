"""MCP manager: connects to user-provided MCP servers and exposes their tools.

mcp.json uses the familiar claude-desktop schema:

    {
      "mcpServers": {
        "weather": {"command": "uvx", "args": ["weather-mcp"]},
        "search":  {"url": "http://localhost:9000/mcp"}
      }
    }

FastMCP's client handles both stdio (command/args) and HTTP (url) transports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import Client

from heyo.graph.agents.base import Tool, ToolKit


def load_mcp_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"mcpServers": {}}
    return json.loads(path.read_text())


class MCPManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client: Client | None = None
        self._tools: list[Any] = []

    @property
    def has_servers(self) -> bool:
        return bool(self.config.get("mcpServers"))

    async def start(self) -> None:
        if not self.has_servers:
            return
        self.client = Client(self.config)
        await self.client.__aenter__()
        self._tools = await self.client.list_tools()

    async def stop(self) -> None:
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    def toolkit(self) -> ToolKit:
        """Expose every MCP tool to the agent loop, namespaced by the MCP tool name."""
        kit = ToolKit()
        for tool in self._tools:
            params = tool.inputSchema or {"type": "object", "properties": {}}

            def make_fn(tool_name: str):
                async def call(**kwargs: Any) -> str:
                    result = await self.client.call_tool(tool_name, kwargs)
                    parts = []
                    for block in result.content:
                        parts.append(getattr(block, "text", None) or str(block))
                    return "\n".join(parts) or "(no output)"

                return call

            kit.tools[tool.name] = Tool(
                name=tool.name,
                description=tool.description or tool.name,
                parameters=params,
                fn=make_fn(tool.name),
            )
        return kit

    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]
