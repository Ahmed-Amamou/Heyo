"""STT backend selection: auto prefers Whisper-if-cached, else falls back to Vosk."""

from __future__ import annotations

import pytest

import heyo.voice_io.stt as stt


def test_auto_falls_back_to_vosk_when_whisper_unavailable(monkeypatch):
    monkeypatch.delenv("HEYO_STT_BACKEND", raising=False)

    def boom(*a, **k):
        raise RuntimeError("faster-whisper small unavailable: no model.bin")

    sentinel = object()
    monkeypatch.setattr(stt, "WhisperTranscriber", boom)
    monkeypatch.setattr(stt, "VoskTranscriber", lambda *a, **k: sentinel)
    assert stt.make_transcriber() is sentinel


def test_auto_uses_whisper_when_available(monkeypatch):
    monkeypatch.delenv("HEYO_STT_BACKEND", raising=False)
    whisper = object()
    monkeypatch.setattr(stt, "WhisperTranscriber", lambda *a, **k: whisper)
    monkeypatch.setattr(stt, "VoskTranscriber",
                        lambda *a, **k: pytest.fail("should not reach Vosk"))
    assert stt.make_transcriber() is whisper


def test_forced_whisper_propagates_failure(monkeypatch):
    monkeypatch.setenv("HEYO_STT_BACKEND", "whisper")

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(stt, "WhisperTranscriber", boom)
    monkeypatch.setattr(stt, "VoskTranscriber",
                        lambda *a, **k: pytest.fail("forced whisper must not fall back"))
    with pytest.raises(RuntimeError):
        stt.make_transcriber()


def test_forced_vosk_skips_whisper(monkeypatch):
    monkeypatch.setenv("HEYO_STT_BACKEND", "vosk")
    vosk = object()
    monkeypatch.setattr(stt, "WhisperTranscriber",
                        lambda *a, **k: pytest.fail("forced vosk must not load whisper"))
    monkeypatch.setattr(stt, "VoskTranscriber", lambda *a, **k: vosk)
    assert stt.make_transcriber() is vosk
