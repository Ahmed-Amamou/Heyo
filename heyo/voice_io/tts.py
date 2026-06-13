"""Text-to-speech with Piper. The voice model is auto-downloaded on first use."""

from __future__ import annotations

import io
import logging
import os
import re
import wave
from pathlib import Path

import httpx

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
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


class Speaker:
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
