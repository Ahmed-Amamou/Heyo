"""Text-to-speech with Piper. Downloads the voice model on first use."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _voice_url_parts(voice: str) -> str:
    # en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium
    lang_region, name, quality = voice.split("-", 2)
    lang = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{name}/{quality}/{voice}"


class Speaker:
    def __init__(self, voice: str, data_dir: Path):
        from piper import PiperVoice

        model_path = data_dir / f"{voice}.onnx"
        if not model_path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            base = f"{HF_BASE}/{_voice_url_parts(voice)}"
            print(f"[voice] downloading piper voice {voice} ...")
            for suffix in (".onnx", ".onnx.json"):
                resp = httpx.get(base + suffix, follow_redirects=True, timeout=300)
                resp.raise_for_status()
                (data_dir / f"{voice}{suffix}").write_bytes(resp.content)
        self.voice = PiperVoice.load(str(model_path))

    def say(self, text: str) -> None:
        if not text.strip():
            return
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize(text, wav)
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        sd.play(audio, samplerate=rate, blocking=True)
