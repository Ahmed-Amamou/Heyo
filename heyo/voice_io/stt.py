"""Speech-to-text: faster-whisper when its model is on disk, else Vosk.

Both backends take raw audio bytes — wav from the python client, webm/opus from
the browser mic — and return text. faster-whisper is more accurate but its
~480MB model lives on HuggingFace; Vosk's small model comes from a different
host (and is tiny), so it's the working default when HF is unreachable. `auto`
prefers an already-cached Whisper and otherwise uses Vosk, so the system
upgrades itself the moment the Whisper model appears.
"""

from __future__ import annotations

import io
import json
import logging
import os

log = logging.getLogger("heyo")


def make_transcriber():
    """Build the configured STT backend.

    HEYO_STT_BACKEND = auto (default) | whisper | vosk
    `auto` loads faster-whisper only if its model is already cached — it never
    triggers a blocking HF download here (that's what hung on throttled links).
    """
    backend = os.getenv("HEYO_STT_BACKEND", "auto").lower()
    if backend in ("auto", "whisper"):
        try:
            return WhisperTranscriber(allow_download=backend == "whisper")
        except Exception as exc:
            if backend == "whisper":
                raise
            log.info("faster-whisper not available (%s); using Vosk for STT", exc)
    return VoskTranscriber()


def _decode_pcm16k(audio: bytes) -> bytes:
    """Any container/codec -> 16 kHz mono signed-16-bit PCM, via PyAV (ffmpeg)."""
    import av

    chunks: list[bytes] = []
    with av.open(io.BytesIO(audio)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().tobytes())
        for out in resampler.resample(None):  # flush the resampler
            chunks.append(out.to_ndarray().tobytes())
    return b"".join(chunks)


class WhisperTranscriber:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 allow_download: bool = False):
        from faster_whisper import WhisperModel

        model_size = model_size or os.getenv("HEYO_STT_MODEL", "small")
        # Prefer the GPU: transcription finishes before the LLM starts generating,
        # so they never compute at once; small-int8_float16 (~0.5GB) sits beside
        # the resident LLM and is much faster than CPU. Falls back to CPU if CUDA
        # isn't usable. HEYO_STT_DEVICE forces cpu/cuda.
        device = device or os.getenv("HEYO_STT_DEVICE", "auto")
        self.language = os.getenv("HEYO_STT_LANGUAGE") or None
        self.model = self._load(WhisperModel, model_size, device, allow_download)

    @staticmethod
    def _load(WhisperModel, model_size: str, device: str, allow_download: bool):
        devices = ["cuda", "cpu"] if device == "auto" else [device]
        # Offline first: a cached model must never phone home to check HF for
        # updates — on a throttled-HF link that call HANGS the whole load. Only
        # attempt an online (downloading) load when explicitly asked. With no
        # network, a missing-CUDA/cuDNN error raises fast -> CPU, never hangs.
        passes = (True, False) if allow_download else (True,)
        last: Exception | None = None
        for local_only in passes:
            for dev in devices:
                compute = "int8" if dev == "cpu" else "int8_float16"
                try:
                    m = WhisperModel(model_size, device=dev, compute_type=compute,
                                     local_files_only=local_only)
                    log.info("faster-whisper %s on %s (%s%s)", model_size, dev, compute,
                             "" if local_only else ", downloaded")
                    return m
                except Exception as exc:  # no CUDA/cuDNN, not cached, OOM, ...
                    last = exc
                    log.warning("faster-whisper %s on %s failed (%s)", model_size, dev, exc)
        raise RuntimeError(f"faster-whisper {model_size} unavailable: {last}")

    def transcribe(self, audio: bytes) -> str:
        segments, _ = self.model.transcribe(
            io.BytesIO(audio), language=self.language, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class VoskTranscriber:
    """Offline STT with Vosk. The small en-us model (~40MB) auto-downloads from
    alphacephei.com (not HuggingFace) on first use, or point HEYO_VOSK_MODEL at
    a model directory."""

    def __init__(self, model_path: str | None = None):
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        model_path = model_path or os.getenv("HEYO_VOSK_MODEL")
        self.model = Model(model_path) if model_path else Model(lang="en-us")
        log.info("vosk STT ready (%s)", model_path or "small en-us")

    def transcribe(self, audio: bytes) -> str:
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self.model, 16000)
        rec.AcceptWaveform(_decode_pcm16k(audio))
        return json.loads(rec.FinalResult()).get("text", "").strip()
