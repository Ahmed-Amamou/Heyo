"""POST /voice/transcribe + /voice/speak — the server's ears and mouth.

STT/TTS run server-side so voice clients (Windows tray client, browser mic)
stay thin. Models load lazily on the first request and stay warm; if the
voice extra isn't installed the endpoints answer 503 with an install hint.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

log = logging.getLogger("heyo")
router = APIRouter(prefix="/voice")

_stt = None
_tts = None
_lock = asyncio.Lock()  # one transcription/synthesis at a time — voice is single-user


def _load_stt():
    global _stt
    if _stt is None:
        from heyo.voice_io.stt import make_transcriber

        log.info("loading speech-to-text (first /voice/transcribe)")
        _stt = make_transcriber()
    return _stt


def _load_tts():
    global _tts
    if _tts is None:
        from heyo.voice_io.tts import make_speaker

        log.info("loading text-to-speech (first /voice/speak)")
        _tts = make_speaker()
    return _tts


async def _get(loader, what: str):
    try:
        return await asyncio.to_thread(loader)
    except ImportError as exc:
        raise HTTPException(
            503, f"{what} unavailable — install server voice deps: uv sync --extra voice ({exc})"
        ) from exc
    except Exception as exc:
        raise HTTPException(503, f"{what} failed to load: {exc}") from exc


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None  # per-request voice override (e.g. picked in the UI)


@router.get("/voices")
async def voices():
    """Available TTS voices + the configured default (for the UI picker)."""
    tts = await _get(_load_tts, "text-to-speech")
    return {"voices": tts.voices(), "current": getattr(tts, "current", None)}


@router.post("/transcribe")
async def transcribe(request: Request):
    """Raw audio bytes in (wav/webm/ogg/...), {"text": ...} out."""
    audio = await request.body()
    if len(audio) < 200:
        raise HTTPException(400, "no audio received")
    stt = await _get(_load_stt, "speech-to-text")
    async with _lock:
        text = await asyncio.to_thread(stt.transcribe, audio)
    return {"text": text}


@router.post("/speak")
async def speak(req: SpeakRequest):
    """{"text": ...} in, WAV bytes out."""
    if not req.text.strip():
        raise HTTPException(400, "no text to speak")
    tts = await _get(_load_tts, "text-to-speech")
    async with _lock:
        wav = await asyncio.to_thread(tts.wav_bytes, req.text, req.voice)
    return Response(content=wav, media_type="audio/wav")
