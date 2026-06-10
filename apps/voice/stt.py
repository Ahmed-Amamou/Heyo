"""Speech-to-text with faster-whisper."""

from __future__ import annotations

import numpy as np


class Transcriber:
    def __init__(self, model_size: str = "small"):
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        except Exception:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language=None, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
