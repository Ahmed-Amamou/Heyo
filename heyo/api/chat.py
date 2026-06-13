"""POST /chat — runs the agent graph, streaming trace + token events over SSE."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    voice: bool = False  # spoken session: bias agents toward short, readable-aloud answers


VOICE_STYLE = (
    "Voice session: the user spoke this and will hear your answer read aloud. "
    "Answer in short plain sentences — no markdown, no lists, no code blocks."
)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    graph = request.app.state.graph
    sessions = request.app.state.sessions
    session_id = req.session_id or str(uuid.uuid4())

    sessions.append(session_id, {"role": "user", "content": req.message})
    messages = list(sessions.history(session_id))
    if req.voice:
        # transient style hint — not persisted, so typed follow-ups aren't affected
        messages.append({"role": "system", "content": VOICE_STYLE})
    state = {"session_id": session_id, "messages": messages}

    async def events():
        final_response = ""
        try:
            async for mode, payload in graph.astream(
                state, stream_mode=["custom", "values"]
            ):
                if mode == "custom":
                    yield {"event": payload.pop("type", "trace"), "data": json.dumps(payload)}
                elif mode == "values" and payload.get("response"):
                    final_response = payload["response"]
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
            return
        if final_response:
            sessions.append(session_id, {"role": "assistant", "content": final_response})
        yield {
            "event": "done",
            "data": json.dumps({"session_id": session_id, "response": final_response}),
        }

    return EventSourceResponse(events())
