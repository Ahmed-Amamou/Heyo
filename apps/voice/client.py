#!/usr/bin/env python3
"""Heyo voice client — a mic and a speaker; the server does the thinking.

Heavy lifting (Whisper STT, Piper TTS, the agents) runs on the Heyo server, so
this client is one file with four small deps and works the same on Windows
(native mic + global hotkey — the seamless setup) and Linux/WSL.

    say "Heyo"  ──or──  press the hotkey (ctrl+alt+h)  ──or──  press Enter here
        → "Yes sir, what can I do for you?"
        → speak your command → the answer streams here and is spoken back

Deps:    pip install numpy sounddevice httpx vosk keyboard
Run:     python client.py [--server http://localhost:8000] [--hotkey ctrl+alt+h]
Windows: see setup.ps1 (installs deps + optional start-at-login)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np

try:
    import sounddevice as sd
except OSError:  # PortAudio missing (e.g. WSL without libportaudio2) — mic I/O
    sd = None    # is unavailable, but wake engines + helpers still import fine

RATE = 16000
FRAME = 1280  # 80 ms — what openWakeWord expects; vosk takes anything
GREETING = os.getenv("HEYO_GREETING", "Yes sir, what can I do for you?")
RETRY_PHRASE = "Sorry, I didn't catch that."
CACHE_DIR = Path.home() / ".heyo" / "client-cache"

DIM, RESET = "\033[2m", "\033[0m"


# ---------------------------------------------------------------- wake engines

class VoskWake:
    """Keyword-spots the literal word "heyo" with a tiny grammar-constrained
    recognizer (vosk small-en, ~40MB, auto-downloaded, CPU-cheap).

    "heyo" isn't an English word, so the grammar pins the recognizer to a few
    tokens and we match the spoken form: "hey yo" / "hey oh"."""

    GRAMMAR = ["hey", "yo", "oh", "hey yo", "hey oh", "you", "hello", "they", "a", "[unk]"]
    PATTERN = re.compile(r"\bhey[\s-]*(yo|oh)\b")
    name = 'vosk — listening for "heyo"'

    def __init__(self):
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.rec = KaldiRecognizer(Model(lang="en-us"), RATE, json.dumps(self.GRAMMAR))
        self.tail = ""  # rolling window of recent finalized words

    def feed(self, frame: np.ndarray) -> bool:
        # "Heyo" decodes to the tokens "hey" + "yo", which vosk's silence
        # endpointing often splits across results — so match over a rolling window
        # of recent finals plus the live partial, not within one frame. Spotting a
        # coined word this way is approximate (the global hotkey is the reliable
        # trigger; a trained openWakeWord model is the robust voice path).
        if self.rec.AcceptWaveform(frame.tobytes()):
            self.tail = f"{self.tail} {json.loads(self.rec.Result()).get('text', '')}".strip()[-40:]
            hay = self.tail
        else:
            partial = json.loads(self.rec.PartialResult()).get("partial", "")
            hay = f"{self.tail} {partial}".strip()
        if self.PATTERN.search(hay):
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.rec.Reset()
        self.tail = ""


class OwwWake:
    """openWakeWord — pretrained 'hey_jarvis' by default, or a custom-trained
    heyo.onnx via HEYO_WAKE_MODEL=/path/to/heyo.onnx."""

    def __init__(self, model_name: str, threshold: float):
        from openwakeword import utils
        from openwakeword.model import Model

        utils.download_models()  # no-op when cached
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self.threshold = threshold
        self.name = f'openwakeword — listening for "{model_name}"'

    def feed(self, frame: np.ndarray) -> bool:
        scores = self.model.predict(frame)
        if any(score >= self.threshold for score in scores.values()):
            self.model.reset()
            return True
        return False

    def reset(self) -> None:
        self.model.reset()


def build_wake(engine: str, model: str, threshold: float):
    candidates = {"vosk": lambda: VoskWake(), "oww": lambda: OwwWake(model, threshold)}
    order = [engine] if engine in candidates else ["vosk", "oww"] if engine == "auto" else []
    for name in order:
        try:
            return candidates[name]()
        except Exception as exc:
            print(f"[heyo] wake engine {name} unavailable: {exc}")
    return None


# ---------------------------------------------------------------- audio helpers

def to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def play_wav(data: bytes, interrupt: threading.Event | None = None) -> None:
    with wave.open(io.BytesIO(data), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    sd.play(audio, samplerate=rate)
    while sd.get_stream().active:
        if interrupt is not None and interrupt.is_set():
            sd.stop()  # barge-in: the hotkey/wake cuts the speech short
            return
        time.sleep(0.05)


def flush(q: queue.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            return


def record_command(q: queue.Queue, max_seconds: float = 15.0,
                   silence_seconds: float = 1.1) -> np.ndarray:
    """Record from the shared mic queue until trailing silence (adaptive RMS)."""
    frames: list[np.ndarray] = []
    silence_needed = int(silence_seconds * RATE / FRAME)
    max_frames = int(max_seconds * RATE / FRAME)
    patience_frames = int(6.0 * RATE / FRAME)  # give up if nothing is ever said
    min_speech = int(0.25 * RATE / FRAME)
    silent = speech = 0
    noise = None
    for i in range(max_frames):
        frame = q.get()
        frames.append(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        noise = rms if noise is None else min(noise * 1.02, max(rms, 1.0), 600.0)
        if rms > max(noise * 3.5, 350.0):
            speech, silent = speech + 1, 0
        elif speech >= min_speech:
            silent += 1
            if silent >= silence_needed:
                break
        elif i > patience_frames:
            return np.zeros(0, dtype=np.int16)
    if speech < min_speech:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames)


# ---------------------------------------------------------------- server link

class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=httpx.Timeout(15, read=600))
        self._warned_tts = False

    def wait_up(self) -> None:
        notice = False
        while True:
            try:
                roles = self.http.get(f"{self.url}/health").json().get("roles", {})
                print(f"[heyo] server up at {self.url} — models: "
                      f"{', '.join(sorted(set(roles.values())))}")
                return
            except Exception:
                if not notice:
                    print(f"[heyo] waiting for the Heyo server at {self.url} ...")
                    notice = True
                time.sleep(5)

    def transcribe(self, wav: bytes) -> str:
        resp = self.http.post(f"{self.url}/voice/transcribe", content=wav,
                              headers={"Content-Type": "audio/wav"})
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    def speak(self, text: str) -> bytes | None:
        try:
            resp = self.http.post(f"{self.url}/voice/speak", json={"text": text})
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as exc:
            if not self._warned_tts:
                print(f"[heyo] server TTS unavailable: {exc.response.text[:200]}")
                self._warned_tts = True
            return None

    def speak_cached(self, text: str) -> bytes | None:
        path = CACHE_DIR / (hashlib.sha1(text.encode()).hexdigest() + ".wav")
        if path.exists():
            return path.read_bytes()
        wav = self.speak(text)
        if wav:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wav)
        return wav

    def chat(self, text: str, session_id: str | None) -> tuple[str, str | None]:
        """POST /chat, print the agent's live stream (thinking dimmed), return the answer."""
        answer, dimmed = "", False
        with self.http.stream("POST", f"{self.url}/chat",
                              json={"message": text, "session_id": session_id,
                                    "voice": True}) as resp:
            resp.raise_for_status()
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = json.loads(line.split(":", 1)[1])
                if event == "thinking":
                    if not dimmed:
                        sys.stdout.write(DIM)
                        dimmed = True
                    sys.stdout.write(data["text"])
                elif event == "trace" and data.get("status") == "tool":
                    sys.stdout.write(f"{RESET}\n{DIM}⌬ {data.get('tool')}"
                                     f"({data.get('args', '')[:80]}){RESET}\n")
                    dimmed = False
                elif event == "token":
                    if dimmed:
                        sys.stdout.write(RESET + "\n")
                        dimmed = False
                    sys.stdout.write(data["text"])
                elif event == "done":
                    answer, session_id = data.get("response", ""), data.get("session_id")
                elif event == "error":
                    answer = f"Something went wrong: {data.get('detail', 'unknown error')}"
                sys.stdout.flush()
        if dimmed:
            sys.stdout.write(RESET)
        print()
        return answer, session_id


# ---------------------------------------------------------------- triggers

def register_hotkey(hotkey: str, trigger: threading.Event) -> bool:
    if not hotkey or hotkey.lower() == "off":
        return False
    try:
        import keyboard
    except ImportError:
        print("[heyo] no global hotkey (pip install keyboard to enable)")
        return False
    try:
        keyboard.add_hotkey(hotkey, trigger.set)
        return True
    except Exception as exc:
        print(f"[heyo] global hotkey unavailable: {exc}")
        return False


def stdin_trigger(trigger: threading.Event) -> None:
    for line in sys.stdin:
        if line.strip().lower() in {"q", "quit", "exit"}:
            os._exit(0)
        trigger.set()


# ---------------------------------------------------------------- main loop

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.getenv("HEYO_API_URL", "http://localhost:8000"))
    p.add_argument("--hotkey", default=os.getenv("HEYO_HOTKEY", "ctrl+alt+h"),
                   help='global push-to-talk hotkey, or "off"')
    p.add_argument("--wake", default=os.getenv("HEYO_WAKE", "auto"),
                   choices=["auto", "vosk", "oww", "off"],
                   help='wake engine: vosk spots the word "heyo"; oww uses openWakeWord')
    p.add_argument("--wake-model", default=os.getenv("HEYO_WAKE_MODEL", "hey_jarvis"),
                   help="openWakeWord model name or path to a custom .onnx")
    p.add_argument("--wake-threshold", type=float,
                   default=float(os.getenv("HEYO_WAKE_THRESHOLD", "0.5")))
    p.add_argument("--device", default=None, help="input device index/name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    if os.name == "nt":
        os.system("")  # enable ANSI colors in the Windows console
    args = parse_args(argv)
    if sd is None:
        print("[heyo] microphone backend unavailable: PortAudio not found.\n"
              "       Linux/WSL: sudo apt install -y libportaudio2\n"
              "       (On Windows the sounddevice wheel bundles PortAudio — no install needed.)")
        return
    if args.list_devices:
        print(sd.query_devices())
        return

    server = Server(args.server)
    server.wait_up()
    wake = None if args.wake == "off" else build_wake(args.wake, args.wake_model,
                                                      args.wake_threshold)
    greeting = server.speak_cached(GREETING)
    retry = server.speak_cached(RETRY_PHRASE)

    trigger = threading.Event()
    hotkey_on = register_hotkey(args.hotkey, trigger)
    threading.Thread(target=stdin_trigger, args=(trigger,), daemon=True).start()

    ways = [w for w in ['say "Heyo"' if wake else None,
                        f"press {args.hotkey}" if hotkey_on else None,
                        "press Enter here"] if w]
    print(f"[heyo] ready — {' · '.join(ways)}  (q + Enter quits)")
    if wake:
        print(f"[heyo] wake engine: {wake.name}")

    q: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, _frames, _time, status):
        if status:
            print(f"[heyo] audio status: {status}")
        q.put(indata.reshape(-1).copy())

    session_id: str | None = None
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        blocksize=FRAME, callback=on_audio, device=device):
        while True:
            if not trigger.is_set():  # wake phase (skipped when barge-in queued one up)
                while not trigger.is_set():
                    try:
                        frame = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if wake and wake.feed(frame):
                        break
            trigger.clear()
            flush(q)
            if greeting:
                play_wav(greeting)
            flush(q)  # drop our own greeting echo before recording
            print("\n[heyo] listening…")
            audio = record_command(q)
            if not len(audio):
                print("[heyo] heard nothing")
                if retry:
                    play_wav(retry)
                continue
            text = server.transcribe(to_wav(audio))
            if not text:
                print("[heyo] couldn't transcribe that")
                if retry:
                    play_wav(retry)
                continue
            print(f"\n› {text}\n")
            try:
                answer, session_id = server.chat(text, session_id)
            except httpx.HTTPError as exc:
                print(f"[heyo] API error: {exc}")
                continue
            wav = server.speak(answer)
            if wav:
                play_wav(wav, interrupt=trigger)  # hotkey/Enter barge-in cuts it short
            if wake:
                wake.reset()
            flush(q)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[heyo] bye")
