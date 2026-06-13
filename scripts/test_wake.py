#!/usr/bin/env python3
"""Wake-word bench: synthesize phrases with Piper, feed them to the wake engine.

Tunes "heyo" detection without touching a microphone:

    uv run python scripts/test_wake.py          # needs --extra voice + voice-client

Reports a detection matrix over positive phrases (must fire) and tricky
negatives (must stay quiet), at three speaking speeds.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.voice.client import FRAME, RATE, VoskWake  # noqa: E402
from heyo.voice_io.tts import Speaker  # noqa: E402

POSITIVE = [
    "Heyo.",
    "heyo",
    "Heyo!",
    "Hey-yo.",
    "Heyo, open the calculator.",
]
NEGATIVE = [
    "Hello there.",
    "Hey, you should see this.",
    "They opened the door.",
    "Yo, what's up?",
    "Hey.",
    "Okay, so let's go.",
    "Are you there?",
]
SPEEDS = (0.8, 1.0, 1.25)


def synth(speaker: Speaker, text: str, length_scale: float) -> bytes:
    buf = io.BytesIO()
    try:
        from piper import SynthesisConfig

        cfg = SynthesisConfig(length_scale=length_scale)
    except Exception:
        cfg = None
    with wave.open(buf, "wb") as w:
        speaker.voice.synthesize_wav(text, w, syn_config=cfg)
    return buf.getvalue()


def to_pcm16k(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate != RATE:
        n = int(len(audio) * RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio), n, endpoint=False),
            np.arange(len(audio)),
            audio.astype(np.float32),
        ).astype(np.int16)
    return audio


def fires(engine: VoskWake, audio: np.ndarray) -> bool:
    pad = np.zeros(RATE // 2, dtype=np.int16)
    audio = np.concatenate([pad, audio, pad])
    hit = False
    for i in range(0, len(audio) - FRAME, FRAME):
        if engine.feed(audio[i : i + FRAME]):
            hit = True
    engine.reset()
    return hit


def main() -> int:
    speaker = Speaker()
    engine = VoskWake()
    misses, false_alarms = [], []
    print(f"{'phrase':<36} " + " ".join(f"x{s}" for s in SPEEDS))
    for expected, phrases in ((True, POSITIVE), (False, NEGATIVE)):
        for phrase in phrases:
            results = []
            for speed in SPEEDS:
                hit = fires(engine, to_pcm16k(synth(speaker, phrase, speed)))
                results.append(hit)
                if expected and not hit:
                    misses.append((phrase, speed))
                if not expected and hit:
                    false_alarms.append((phrase, speed))
            marks = " ".join("✓" if r else "·" for r in results)
            tag = "POS" if expected else "NEG"
            print(f"[{tag}] {phrase:<30} {marks}")
    print(f"\nmissed wakes: {len(misses)}/{len(POSITIVE) * len(SPEEDS)}  "
          f"false alarms: {len(false_alarms)}/{len(NEGATIVE) * len(SPEEDS)}")
    for phrase, speed in misses:
        print(f"  miss: {phrase!r} @x{speed}")
    for phrase, speed in false_alarms:
        print(f"  FALSE ALARM: {phrase!r} @x{speed}")
    return 1 if (misses or false_alarms) else 0


if __name__ == "__main__":
    raise SystemExit(main())
