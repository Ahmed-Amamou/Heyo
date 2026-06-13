"""Speech-to-text with faster-whisper, decoding whatever container the client sent."""

from __future__ import annotations

import io
import os


class Transcriber:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        from faster_whisper import WhisperModel

        model_size = model_size or os.getenv("HEYO_STT_MODEL", "small")
        # cpu by default: the LLM owns the (6GB) GPU, and small-int8 on CPU is
        # fast enough for short spoken commands. HEYO_STT_DEVICE=cuda to override.
        device = device or os.getenv("HEYO_STT_DEVICE", "cpu")
        compute = "int8" if device == "cpu" else "int8_float16"
        self.language = os.getenv("HEYO_STT_LANGUAGE") or None
        self.model = WhisperModel(model_size, device=device, compute_type=compute)

    def transcribe(self, audio: bytes) -> str:
        # faster-whisper decodes via PyAV: wav from the python client,
        # webm/opus from the browser mic — anything ffmpeg-shaped works.
        segments, _ = self.model.transcribe(
            io.BytesIO(audio), language=self.language, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
