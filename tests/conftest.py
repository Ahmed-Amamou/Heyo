from __future__ import annotations

import json
from typing import Any

import pytest

from heyo.config import ModelsConfig, RoleConfig
from heyo.llm.client import LLMClient


class FakeLLM(LLMClient):
    """LLMClient with canned responses instead of HTTP calls.

    - `route_to`: what the router should pick
    - `replies`: assistant messages returned by chat() in order (after routing);
      a dict with "tool_calls" simulates a tool-calling turn
    - `stream_text`: tokens emitted by stream()
    """

    def __init__(self, route_to: str = "chat", replies: list[dict] | None = None,
                 stream_text: str = "hello from fake llm"):
        models = ModelsConfig(roles={"general": RoleConfig(model="fake"),
                                     "router": RoleConfig(model="fake")})
        super().__init__(models)
        self.route_to = route_to
        self.replies = replies or [{"content": "done"}]
        self.stream_text = stream_text
        self.calls: list[dict[str, Any]] = []

    async def chat(self, role, messages, tools=None, temperature=0.2, json_schema=None,
                   think=True):
        self.calls.append({"role": role, "messages": messages, "tools": tools, "think": think})
        if role == "router" or json_schema is not None:
            return {"content": json.dumps({"route": self.route_to, "rationale": "test"})}
        return self.replies.pop(0) if self.replies else {"content": "done"}

    async def stream(self, role, messages, temperature=0.4):
        self.calls.append({"role": role, "messages": messages, "stream": True})
        for tok in self.stream_text.split(" "):
            yield tok + " "


@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "workspace"
