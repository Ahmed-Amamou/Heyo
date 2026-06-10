from __future__ import annotations

import json

from heyo.graph.agents.base import ToolKit, parse_soft_tool_call


def make_kit() -> ToolKit:
    kit = ToolKit()

    @kit.add("write_file", "write", {"type": "object", "properties": {}})
    def write_file(path: str, content: str) -> str:
        return "ok"

    return kit


def test_parses_fenced_json_tool_call():
    content = '```json\n{"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}}\n```'
    assert parse_soft_tool_call(content, make_kit()) == (
        "write_file", {"path": "a.txt", "content": "x"}
    )


def test_parses_bare_json_with_tool_key():
    content = 'Sure, I will do that:\n{"tool": "write_file", "parameters": {"path": "b.txt"}}'
    assert parse_soft_tool_call(content, make_kit()) == ("write_file", {"path": "b.txt"})


def test_multiple_json_objects_takes_first_valid():
    content = ('{"name": "goto", "arguments": {"url": "https://x.com"}}\n'
               '{"name": "read_page", "arguments": {}}')
    kit = ToolKit()

    @kit.add("goto", "nav", {"type": "object", "properties": {}})
    def goto(url: str) -> str:
        return "ok"

    @kit.add("read_page", "read", {"type": "object", "properties": {}})
    def read_page() -> str:
        return "ok"

    assert parse_soft_tool_call(content, kit) == ("goto", {"url": "https://x.com"})


def test_unwraps_schema_shaped_arguments():
    content = '{"name": "write_file", "arguments": {"path": {"type": "string", "value": "a.txt"}, "content": "x"}}'
    assert parse_soft_tool_call(content, make_kit()) == (
        "write_file", {"path": "a.txt", "content": "x"}
    )


def test_ignores_plain_text_and_unknown_tools():
    kit = make_kit()
    assert parse_soft_tool_call("I created the file for you.", kit) is None
    assert parse_soft_tool_call(json.dumps({"name": "rm_rf", "arguments": {}}), kit) is None
    assert parse_soft_tool_call('{"name": "write_file", "arguments": "oops"}', kit) is None
