# Heyo voice client

Always-listening loop: **"Heyo"** (wake word) → *"Yes sir, what can I do for you?"* →
your spoken command → answer spoken back.

## Run (WSL2)

```bash
sudo apt install -y libportaudio2          # one-time, audio backend
uv sync --extra voice
uv run python -m apps.voice.main
```

WSLg exposes the Windows mic/speakers through PulseAudio. Check with
`pactl list sources short` — if no source shows up, use the Windows fallback below.

## Windows-native fallback

The client is plain Python talking to the server over HTTP, so it runs unchanged on
Windows when WSL audio is flaky:

```powershell
py -3.12 -m pip install openwakeword faster-whisper piper-tts sounddevice numpy onnxruntime httpx
$env:HEYO_API_URL = "http://localhost:8000"   # WSL ports are forwarded automatically
py -3.12 -m apps.voice.main
```

## Custom "heyo" wake word

v1 listens for the bundled pretrained **hey_jarvis** model. To react to "Heyo"
literally, train a custom model with openWakeWord's synthetic-speech notebook
(https://github.com/dscripka/openWakeWord#training-new-models), drop the resulting
`heyo.onnx` somewhere, and set:

```bash
HEYO_WAKE_MODEL=/path/to/heyo.onnx
```

## Config (env vars)

| var | default | meaning |
|---|---|---|
| `HEYO_API_URL` | `http://localhost:8000` | Heyo server |
| `HEYO_WAKE_MODEL` | `hey_jarvis` | openWakeWord model name or path to .onnx |
| `HEYO_WAKE_THRESHOLD` | `0.5` | wake sensitivity (lower = more sensitive) |
| `HEYO_WHISPER_MODEL` | `small` | faster-whisper size (tiny/base/small/medium) |
| `HEYO_PIPER_VOICE` | `en_US-lessac-medium` | Piper voice (auto-downloaded) |
| `HEYO_GREETING` | `Yes sir, what can I do for you?` | wake response |
