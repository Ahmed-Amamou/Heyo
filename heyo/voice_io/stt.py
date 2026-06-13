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
from pathlib import Path

log = logging.getLogger("heyo")

# A manually-placed CT2 model lands here (HuggingFace is often throttled, so the
# weights get fetched out-of-band); used automatically when present.
LOCAL_WHISPER = Path.home() / ".heyo" / "whisper-small"


_cuda_preloaded = False


def _preload_cuda_libs() -> None:
    """Make the GPU work no matter how the server was launched (no reliance on
    LD_LIBRARY_PATH): dlopen the CUDA math libs with RTLD_GLOBAL so CTranslate2's
    later dlopen-by-soname resolves to the already-loaded copy. Ollama bundles
    libcublas/libcudart for CUDA 12; the nvidia-*-cu12 wheels are used too if
    installed. Loaded deps-first; silent no-op when nothing is found (→ CPU)."""
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True
    import ctypes
    import glob
    import sys

    dirs = ["/usr/local/lib/ollama/cuda_v12",
            *glob.glob(f"{sys.prefix}/lib/python*/site-packages/nvidia/*/lib")]
    loaded = []
    for soname in ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"):
        for d in dirs:
            hits = glob.glob(f"{d}/{soname}*")
            if not hits:
                continue
            try:
                ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                loaded.append(soname)
            except OSError as exc:
                log.debug("preload %s failed: %s", hits[0], exc)
            break
    if loaded:
        log.info("preloaded CUDA libs for STT: %s", ", ".join(loaded))


def _resolve_model(model_size: str | None) -> str:
    if model_size:
        return model_size
    env = os.getenv("HEYO_STT_MODEL")
    if env:
        return env
    if (LOCAL_WHISPER / "model.bin").exists():
        return str(LOCAL_WHISPER)
    return "small"


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

        model_size = _resolve_model(model_size)
        # Prefer the GPU: transcription finishes before the LLM starts generating,
        # so they never compute at once; small-int8_float16 (~0.5GB) sits beside
        # the resident LLM and is much faster than CPU. Falls back to CPU if the
        # CUDA libs aren't usable. HEYO_STT_DEVICE forces cpu/cuda.
        device = device or os.getenv("HEYO_STT_DEVICE", "auto")
        self.language = os.getenv("HEYO_STT_LANGUAGE") or None
        # beam_size=1 is ~2x faster than the default 5 with no accuracy loss on
        # short commands — the difference between ~5s and ~2.5s on CPU.
        self.beam_size = int(os.getenv("HEYO_STT_BEAM", "1"))
        self.model = self._load(WhisperModel, model_size, device, allow_download)

    @staticmethod
    def _load(WhisperModel, model_size: str, device: str, allow_download: bool):
        import numpy as np

        devices = ["cuda", "cpu"] if device == "auto" else [device]
        if "cuda" in devices:
            _preload_cuda_libs()  # so the GPU works without LD_LIBRARY_PATH
        # Offline first: a cached model must never phone home to check HF for
        # updates — on a throttled-HF link that call HANGS the whole load. Only
        # attempt an online (downloading) load when explicitly asked.
        passes = (True, False) if allow_download else (True,)
        # A GPU can *load* the model yet fail at inference for missing CUDA math
        # libs (libcublas/libcudnn) — so we don't trust a device until a tiny
        # warm-up run actually executes on it; otherwise fall through to CPU.
        warmup = (np.random.randn(16000) * 1e-3).astype(np.float32)
        last: Exception | None = None
        for local_only in passes:
            for dev in devices:
                compute = "int8" if dev == "cpu" else "int8_float16"
                try:
                    m = WhisperModel(model_size, device=dev, compute_type=compute,
                                     cpu_threads=os.cpu_count() or 4,
                                     local_files_only=local_only)
                    list(m.transcribe(warmup, language="en", vad_filter=False, beam_size=1)[0])
                    log.info("faster-whisper %s on %s (%s%s)", model_size, dev, compute,
                             "" if local_only else ", downloaded")
                    return m
                except Exception as exc:  # not cached, no CUDA libs, OOM, ...
                    last = exc
                    log.warning("faster-whisper %s on %s unusable (%s)", model_size, dev, exc)
        raise RuntimeError(f"faster-whisper {model_size} unavailable: {last}")

    def transcribe(self, audio: bytes) -> str:
        segments, _ = self.model.transcribe(
            io.BytesIO(audio), language=self.language,
            beam_size=self.beam_size, vad_filter=True,
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
