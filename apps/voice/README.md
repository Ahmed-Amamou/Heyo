# Heyo voice client

A mic and a speaker — **the server does the thinking, the hearing and the talking**
(Whisper STT and Piper TTS run server-side behind `/voice/transcribe` and
`/voice/speak`). The client is one file, four small deps, identical on Windows
and Linux.

```
say "Heyo"   ──or──   press ctrl+alt+h (global)   ──or──   press Enter in its console
    → "Yes sir, what can I do for you?"
    → speak your command
    → the agent's reasoning streams in the console, the answer is spoken back
```

Pressing the hotkey while Heyo is speaking cuts it off and listens again (barge-in).

## Windows — the seamless setup (recommended)

The Heyo server runs in WSL; WSL forwards `localhost`, so the client runs natively
on Windows with first-class mic access and a true global hotkey. In PowerShell:

```powershell
cd \\wsl.localhost\Ubuntu\home\hamoud\Heyo\apps\voice
powershell -ExecutionPolicy Bypass -File setup.ps1 -Startup
```

That installs the deps, copies the client to `%LOCALAPPDATA%\Heyo`, registers it to
start (hidden) at login, and launches it. From then on: just say **"Heyo"** anywhere
in Windows, or hit **ctrl+alt+h**. Drop `-Startup` to try it without the autostart.

## Linux / WSL

```bash
sudo apt install -y libportaudio2     # one-time audio backend
uv sync --extra voice-client
uv run python apps/voice/client.py
```

(WSLg microphone capture can be flaky — that's exactly why the Windows-native
path above exists.)

## How "Heyo" is detected

| engine | wake phrase | notes |
|---|---|---|
| `vosk` (default) | **"Heyo"** | tiny grammar-constrained recognizer spots "hey-yo"/"hey-oh"; ~40MB model auto-downloads on first run |
| `oww` | "hey jarvis" (or custom) | openWakeWord; train a real `heyo.onnx` ([guide](https://github.com/dscripka/openWakeWord#training-new-models)) and point `--wake-model` at it |
| `off` | — | hotkey / Enter only |

Pick one with `--wake vosk|oww|off`; `auto` tries vosk, then oww.

## Options (flags or env vars)

| flag | env | default | meaning |
|---|---|---|---|
| `--server` | `HEYO_API_URL` | `http://localhost:8000` | Heyo server |
| `--hotkey` | `HEYO_HOTKEY` | `ctrl+alt+h` | global push-to-talk (`off` to disable) |
| `--wake` | `HEYO_WAKE` | `auto` | wake engine (see above) |
| `--wake-model` | `HEYO_WAKE_MODEL` | `hey_jarvis` | openWakeWord model name or .onnx path |
| `--wake-threshold` | `HEYO_WAKE_THRESHOLD` | `0.5` | openWakeWord sensitivity |
| `--device` | — | system default | mic device (`--list-devices` to inspect) |
| — | `HEYO_GREETING` | `Yes sir, what can I do for you?` | wake response |

Server-side voice knobs (set where the API runs): `HEYO_STT_MODEL` (whisper size,
default `small`), `HEYO_STT_DEVICE` (`auto` — GPU if usable, else CPU; transcription
runs before the LLM generates, so they don't compete), `HEYO_STT_LANGUAGE` (default
auto). TTS is **Kokoro-82M** by default (natural voice; model from GitHub so it
works when HF is throttled; runs on CPU): `HEYO_TTS_VOICE` (default `af_heart`;
also `af_bella`, `am_michael`, `bf_emma`, …), `HEYO_TTS_SPEED`. `HEYO_TTS_ENGINE=piper`
falls back to Piper (`HEYO_PIPER_VOICE`, `HEYO_PIPER_BASE_URL`).
