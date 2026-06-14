"""Text-to-speech. Kokoro-82M by default (natural voice), Piper as fallback.

Both auto-download their model on first use. Kokoro's model + voices come from
GitHub releases (not HuggingFace), so they fetch fast even when HF is throttled,
and it runs on CPU via onnxruntime — leaving the GPU for STT and the LLM.
"""

from __future__ import annotations

import io
import logging
import os
import re
import wave
from pathlib import Path

import httpx

# HuggingFace can be throttled on some links; HEYO_PIPER_BASE_URL points the
# downloader at any mirror with the same layout (e.g. https://hf-mirror.com/...).
# Already have the .onnx + .onnx.json in the voice dir? They're used as-is, no fetch.
HF_BASE = os.getenv(
    "HEYO_PIPER_BASE_URL", "https://huggingface.co/rhasspy/piper-voices/resolve/main"
).rstrip("/")
# Kokoro model + voices live on GitHub releases (fast even when HF is throttled).
KOKORO_BASE = os.getenv(
    "HEYO_KOKORO_BASE_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0",
).rstrip("/")
# Default to the float32 model: on CPUs without AVX-512 VNNI (most consumer chips)
# int8 ops are emulated and ~6x SLOWER, and float32 sounds better anyway. Set
# HEYO_KOKORO_MODEL=kokoro-v1.0.int8.onnx for the smaller model on a VNNI CPU.
KOKORO_MODEL = os.getenv("HEYO_KOKORO_MODEL", "kokoro-v1.0.onnx")
KOKORO_VOICES = "voices-v1.0.bin"
log = logging.getLogger("heyo")


def _voice_url_parts(voice: str) -> str:
    # en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium
    lang_region, name, quality = voice.split("-", 2)
    lang = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{name}/{quality}/{voice}"


def strip_markdown(text: str) -> str:
    """Spoken replies: drop the typography, keep the words."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"```.*?(```|$)", " — code on screen — ", text, flags=re.S)
    text = re.sub(r"`([^`\n]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_>|]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _download(url: str, dest: Path, attempts: int = 8) -> None:
    """Resumable, retrying download. Voice models are tens of MB and residential
    links to HF stall or drop mid-stream; we append to a .part file with a Range
    header so each retry picks up where the last left off instead of restarting
    (and instead of just returning a 503 to the caller)."""
    part = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, follow_redirects=True, headers=headers,
                              timeout=httpx.Timeout(30, read=60)) as resp:
                if resp.status_code == 416:  # already have the whole file
                    break
                if have and resp.status_code == 200:
                    have = 0  # server ignored Range — restart cleanly
                resp.raise_for_status()
                total = have + int(resp.headers.get("content-length", 0))
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    for chunk in resp.iter_bytes(1 << 16):
                        f.write(chunk)
            if not total or part.stat().st_size >= total:
                break
            raise OSError(f"short read ({part.stat().st_size}/{total} bytes)")
        except (httpx.HTTPError, OSError) as exc:
            last_exc = exc
            log.warning("download %s attempt %d/%d failed (%s); resuming",
                        dest.name, attempt, attempts, exc)
    else:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url} after {attempts} tries: {last_exc}")
    part.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def make_speaker(data_dir: Path | None = None):
    """Build the configured TTS engine. HEYO_TTS_ENGINE = kokoro (default) | piper.
    Kokoro sounds far more natural; Piper is the lightweight fallback if Kokoro
    can't load (e.g. its deps aren't installed)."""
    engine = os.getenv("HEYO_TTS_ENGINE", "kokoro").lower()
    if engine != "piper":
        try:
            return KokoroSpeaker(data_dir=data_dir)
        except Exception as exc:
            if engine == "kokoro":
                log.warning("kokoro TTS unavailable (%s); falling back to Piper", exc)
    return PiperSpeaker(data_dir=data_dir)


class KokoroSpeaker:
    """Kokoro-82M (ONNX) — warm, natural speech. Model/voices auto-download from
    GitHub on first use; runs on CPU. HEYO_TTS_VOICE picks the voice (e.g.
    af_heart, af_bella, am_michael, bf_emma); HEYO_TTS_SPEED adjusts pace."""

    def __init__(self, voice: str | None = None, data_dir: Path | None = None):
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        self.voice = voice or os.getenv("HEYO_TTS_VOICE", "af_heart")
        self.speed = float(os.getenv("HEYO_TTS_SPEED", "1.0"))
        self.lang = os.getenv("HEYO_TTS_LANG", "en-us")
        data_dir = Path(data_dir or os.getenv("HEYO_VOICE_DATA", "~/.heyo/voice")).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        model_path = data_dir / KOKORO_MODEL
        voices_path = data_dir / KOKORO_VOICES
        if not model_path.exists():
            _download(f"{KOKORO_BASE}/{KOKORO_MODEL}", model_path)
        if not voices_path.exists():
            _download(f"{KOKORO_BASE}/{KOKORO_VOICES}", voices_path)
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        # kokoro-onnx builds a default-threaded session; swap in one that uses
        # every core with full graph optimization — TTS is CPU-bound here.
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.kokoro.sess = rt.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        log.info("kokoro TTS ready (voice=%s)", self.voice)

    def wav_bytes(self, text: str, max_chars: int = 2000) -> bytes:
        import numpy as np

        text = strip_markdown(text)[:max_chars] or "Done."
        samples, sr = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())
        return buf.getvalue()


class PiperSpeaker:
    def __init__(self, voice: str | None = None, data_dir: Path | None = None):
        from piper import PiperVoice

        voice = voice or os.getenv("HEYO_PIPER_VOICE", "en_US-lessac-medium")
        data_dir = Path(
            data_dir or os.getenv("HEYO_VOICE_DATA", "~/.heyo/voice")
        ).expanduser()
        model_path = data_dir / f"{voice}.onnx"
        if not model_path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            base = f"{HF_BASE}/{_voice_url_parts(voice)}"
            for suffix in (".onnx", ".onnx.json"):
                _download(base + suffix, data_dir / f"{voice}{suffix}")
        self.voice = PiperVoice.load(str(model_path))

    def wav_bytes(self, text: str, max_chars: int = 1200) -> bytes:
        text = strip_markdown(text)[:max_chars] or "Done."
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize_wav(text, wav)
        return buf.getvalue()


# Back-compat alias (older imports expected `Speaker`).
Speaker = PiperSpeaker
