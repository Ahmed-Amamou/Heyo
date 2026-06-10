"""Voice client configuration (env-driven, shares .env with the server)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VoiceConfig:
    api_url: str = os.getenv("HEYO_API_URL", "http://localhost:8000")
    # openWakeWord pretrained model to react to; "hey_jarvis" is the stand-in
    # until a custom "heyo" model is trained (drop its .onnx path here).
    wake_model: str = os.getenv("HEYO_WAKE_MODEL", "hey_jarvis")
    wake_threshold: float = float(os.getenv("HEYO_WAKE_THRESHOLD", "0.5"))
    whisper_model: str = os.getenv("HEYO_WHISPER_MODEL", "small")
    piper_voice: str = os.getenv("HEYO_PIPER_VOICE", "en_US-lessac-medium")
    sample_rate: int = 16000
    frame_samples: int = 1280  # 80 ms frames, what openWakeWord expects
    greeting: str = os.getenv("HEYO_GREETING", "Yes sir, what can I do for you?")
    # command recording: stop after this much trailing silence, cap total length
    silence_seconds: float = 1.2
    max_command_seconds: float = 15.0
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("HEYO_VOICE_DATA", "~/.heyo/voice")).expanduser()
    )
