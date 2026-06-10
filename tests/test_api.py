from __future__ import annotations

import json

import httpx
import pytest
from asgi_lifespan import LifespanManager

pytest.importorskip("asgi_lifespan")

from heyo.api.chat import router as chat_router  # noqa: E402
from heyo.api.sessions import SessionStore  # noqa: E402
from heyo.graph.build import build_graph  # noqa: E402

from tests.conftest import FakeLLM  # noqa: E402


def make_test_app(workspace):
    """App with the real /chat route but a fake LLM and no Qdrant/MCP/lifespan deps."""
    from fastapi import FastAPI

    from heyo.config import Settings

    app = FastAPI()
    app.include_router(chat_router)
    app.state.sessions = SessionStore()
    app.state.graph = build_graph(
        FakeLLM(route_to="chat", stream_text="hello world"),
        Settings(heyo_workspace=workspace),
    )
    return app


async def test_chat_endpoint_streams_sse(workspace):
    app = make_test_app(workspace)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            events = []
            async with client.stream(
                "POST", "/chat", json={"message": "hi"}
            ) as resp:
                assert resp.status_code == 200
                event = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event:
                        events.append((event, json.loads(line.split(":", 1)[1])))

    kinds = [e for e, _ in events]
    assert "trace" in kinds and "token" in kinds and kinds[-1] == "done"
    done = events[-1][1]
    assert done["response"] == "hello world"
    assert done["session_id"]
    # session history retained for follow-ups
    hist = app.state.sessions.history(done["session_id"])
    assert [m["role"] for m in hist] == ["user", "assistant"]
