"""Heyo voice client: always-listening wake loop.

    say "Heyo" (wake word) -> "Yes sir, what can I do for you?"
    -> speak a command -> transcribed -> sent to the Heyo API -> answer spoken back

Run with:  uv run --extra voice python -m apps.voice.main
Works in WSL2 via WSLg/PulseAudio; if mic capture is unreliable there, run this same
client natively on Windows pointing HEYO_API_URL at the WSL server.
"""

from __future__ import annotations

import json
import queue

import httpx
import numpy as np
import sounddevice as sd

from apps.voice.config import VoiceConfig
from apps.voice.stt import Transcriber
from apps.voice.tts import Speaker
from apps.voice.wake import WakeDetector


def record_command(cfg: VoiceConfig) -> np.ndarray:
    """Record until trailing silence (RMS endpointing) or the max-length cap."""
    frames: list[np.ndarray] = []
    silence_frames_needed = int(cfg.silence_seconds * cfg.sample_rate / cfg.frame_samples)
    max_frames = int(cfg.max_command_seconds * cfg.sample_rate / cfg.frame_samples)
    silent, heard_speech = 0, False
    noise_floor = None

    with sd.InputStream(
        samplerate=cfg.sample_rate, channels=1, dtype="int16", blocksize=cfg.frame_samples
    ) as stream:
        for _ in range(max_frames):
            frame, _ = stream.read(cfg.frame_samples)
            frame = frame.reshape(-1)
            frames.append(frame)
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            noise_floor = rms if noise_floor is None else min(noise_floor * 1.01, rms, 500.0)
            speaking = rms > max(noise_floor * 3, 300.0)
            if speaking:
                heard_speech, silent = True, 0
            elif heard_speech:
                silent += 1
                if silent >= silence_frames_needed:
                    break
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def ask_heyo(cfg: VoiceConfig, text: str, session_id: str | None) -> tuple[str, str | None]:
    """POST to /chat and collect the final response from the SSE stream."""
    response, sid = "", session_id
    with httpx.stream(
        "POST",
        f"{cfg.api_url}/chat",
        json={"message": text, "session_id": session_id},
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "done":
                data = json.loads(line.split(":", 1)[1])
                response, sid = data.get("response", ""), data.get("session_id", sid)
    return response, sid


def main() -> None:
    cfg = VoiceConfig()
    print(f"[voice] loading models (wake={cfg.wake_model}, whisper={cfg.whisper_model}) ...")
    wake = WakeDetector(cfg.wake_model, cfg.wake_threshold)
    stt = Transcriber(cfg.whisper_model)
    tts = Speaker(cfg.piper_voice, cfg.data_dir)
    session_id: str | None = None

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, _frames, _time, status):
        if status:
            print(f"[voice] audio status: {status}")
        audio_q.put(indata.reshape(-1).copy())

    print(f"[voice] listening for the wake word — say it and ask away. API: {cfg.api_url}")
    while True:
        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=cfg.frame_samples,
            callback=on_audio,
        ):
            woke = False
            while not woke:
                woke = wake.feed(audio_q.get())
        # stream is closed here so the mic is free for command recording
        print("[voice] wake word detected")
        tts.say(cfg.greeting)
        audio = record_command(cfg)
        text = stt.transcribe(audio, cfg.sample_rate)
        if not text:
            tts.say("Sorry, I didn't catch that.")
            continue
        print(f"[voice] you said: {text}")
        try:
            answer, session_id = ask_heyo(cfg, text, session_id)
        except httpx.HTTPError as exc:
            tts.say("I couldn't reach the Heyo server.")
            print(f"[voice] API error: {exc}")
            continue
        print(f"[voice] heyo: {answer}")
        tts.say(answer[:600])


if __name__ == "__main__":
    main()
