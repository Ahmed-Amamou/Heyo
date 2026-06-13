"""/voice endpoints with the heavy STT/TTS models faked out."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

import heyo.api.voice as voice_api
from heyo.voice_io.tts import strip_markdown


class FakeSTT:
    def transcribe(self, audio: bytes) -> str:
        return f"heard {len(audio)} bytes"


class FakeTTS:
    def wav_bytes(self, text: str) -> bytes:
        return b"RIFF-fake-wav:" + text.encode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(voice_api, "_stt", FakeSTT())
    monkeypatch.setattr(voice_api, "_tts", FakeTTS())
    app = FastAPI()
    app.include_router(voice_api.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t")


async def test_transcribe_returns_text(client):
    async with client:
        resp = await client.post("/voice/transcribe", content=b"x" * 1000,
                                 headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "heard 1000 bytes"


async def test_transcribe_rejects_empty_body(client):
    async with client:
        resp = await client.post("/voice/transcribe", content=b"")
    assert resp.status_code == 400


async def test_speak_returns_wav(client):
    async with client:
        resp = await client.post("/voice/speak", json={"text": "hello there"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFF-fake-wav:hello there"


async def test_speak_503_when_voice_extra_missing(monkeypatch):
    monkeypatch.setattr(voice_api, "_tts", None)
    monkeypatch.setattr(voice_api, "_load_tts",
                        lambda: (_ for _ in ()).throw(ImportError("no piper")))
    app = FastAPI()
    app.include_router(voice_api.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/voice/speak", json={"text": "hi"})
    assert resp.status_code == 503
    assert "uv sync --extra voice" in resp.json()["detail"]


def test_strip_markdown_keeps_the_words():
    text = ("## Result\nThe answer is **42** — see [docs](https://x.y).\n"
            "```python\nprint(42)\n```\nDone `inline` _here_."
            "<think>secret</think>")
    spoken = strip_markdown(text)
    assert "42" in spoken and "docs" in spoken and "inline" in spoken
    assert "**" not in spoken and "```" not in spoken and "https://" not in spoken
    assert "#" not in spoken and "secret" not in spoken
