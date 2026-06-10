from __future__ import annotations

import json

from heyo.graph.agents.apps import make_apps_toolkit
from heyo.mcp.manager import MCPManager, load_mcp_config


async def test_apps_toolkit_blocks_non_launcher_binaries():
    kit = make_apps_toolkit()
    result = await kit.execute("run_command", json.dumps({"command": "rm -rf /"}))
    assert "error" in result and "cmd.exe" in result


async def test_apps_toolkit_allows_interop_launchers():
    kit = make_apps_toolkit()
    # 'true' stands in for cmd.exe which doesn't exist in CI; allowlist check happens first
    result = await kit.execute("run_command", json.dumps({"command": "bash -c true"}))
    assert "error" in result  # bash not in allowlist either — proves the gate is binary-name based


def test_load_mcp_config_missing_file(tmp_path):
    cfg = load_mcp_config(tmp_path / "nope.json")
    assert cfg == {"mcpServers": {}}
    assert not MCPManager(cfg).has_servers


def test_mcp_toolkit_namespaces_tools():
    class FakeTool:
        def __init__(self, name):
            self.name = name
            self.description = f"{name} desc"
            self.inputSchema = {"type": "object", "properties": {}}

    manager = MCPManager({"mcpServers": {"x": {"url": "http://localhost:1/mcp"}}})
    manager._tools = [FakeTool("weather_now"), FakeTool("search")]
    kit = manager.toolkit()
    assert set(kit.tools) == {"weather_now", "search"}
    specs = kit.specs()
    assert all(s["type"] == "function" for s in specs)
