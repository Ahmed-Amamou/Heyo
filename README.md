# Heyo

**On-premise multi-LLM agentic assistant.** Say *"Heyo"* — it answers *"Yes sir, what can
I do for you?"* — then routes your request through a LangGraph agent graph running
entirely on your machine: local models (Ollama / vLLM), local memory (Qdrant), local
voice (openWakeWord / faster-whisper / Piper), and your own MCP servers.

```
 voice client (Windows-native     FastAPI server
 or Linux — mic+speaker only)     /voice/transcribe (Whisper) · /voice/speak (Piper)
 "Heyo" / ctrl+alt+h ──HTTP──▶   POST /chat (SSE: trace/thinking/token/done)
 ◀── spoken reply                 web console at / (chat + inline run-chain + mic)
                                          │
                                 LangGraph StateGraph
                       prepare ─▶ router (structured routing)
                  ┌─────────┬─────────┼─────────┬─────────┐
                 chat     files      web      apps      mcp
                (talk)  (workspace)(Playwright)(Windows  (your MCP
                                              interop)   servers)
                  └─────────┴─────────┼─────────┴─────────┘
                                  finalize ◀─▶ Qdrant (memory + skills)
                                          │
                              Ollama / vLLM (OpenAI-compatible)
```

## Features

- **Multi-agent orchestration** — a supervisor LLM routes each request (with a visible
  rationale) to specialized agents over LangGraph conditional edges.
- **Multi-LLM, hardware-aware** — every agent role (router, general, coder, embedder)
  maps to a model+backend in `models.yaml`; `scripts/fit_models.py` uses
  [llmfit](https://github.com/AlexsJones/llmfit) to pick models that actually fit your
  GPU. Ollama by default, vLLM via a Docker profile for bigger hardware.
- **Teachable skills** — drop a markdown file in `skills/` (frontmatter + instructions)
  and Heyo embeds it in Qdrant; relevant skills are retrieved per request and injected
  into the agent's prompt. That's how you teach it *your* app-launch commands, recipes,
  conventions. Live reload: `POST /skills/reload`.
- **Long-term memory** — every exchange is embedded into Qdrant and recalled by
  similarity on later requests, across sessions.
- **MCP integration** — declare your servers in `mcp.json` (claude-desktop schema,
  stdio or HTTP); their tools are exposed to a dedicated agent via FastMCP.
- **Voice, seamless from Windows** — say *"Heyo"* anywhere (or hit a global hotkey):
  STT/TTS run server-side (`/voice/transcribe`, `/voice/speak`), so the always-listening
  client is one file with four small deps that runs natively on Windows while the brain
  stays in WSL. Barge-in, login autostart, browser mic too.
  See [apps/voice/README.md](apps/voice/README.md).
- **A console that shows the agent working** — minimalist web UI: just the name up top
  (the "o" is a status orb tinted by whichever agent is active), and every reply carries
  its own inline run-chain — route → live-streamed thinking → tool calls → answer —
  folding into a one-line summary when done.

## Quick start

```bash
./scripts/setup.sh                     # uv + ollama + llmfit + python deps
uv run python scripts/fit_models.py   # hardware-fit check → pulls models → models.yaml
docker compose up -d qdrant           # vector store (memory + skills)
uv run heyo-api                       # server + web console on http://localhost:8000
```

Voice (optional). Server side, one-time: `uv sync --extra voice`. Then the client —
**from Windows** (recommended: native mic + global hotkey, WSL forwards localhost):

```powershell
cd \\wsl.localhost\Ubuntu\home\<you>\Heyo\apps\voice
powershell -ExecutionPolicy Bypass -File setup.ps1 -Startup   # say "Heyo" or ctrl+alt+h
```

or inside Linux/WSL: `sudo apt install -y libportaudio2 && uv sync --extra voice-client
&& uv run python apps/voice/client.py`

Web agent (optional): `uv sync --extra web && uv run playwright install chromium`

## Teaching Heyo a skill

`skills/open-spotify.md`:

```markdown
---
name: open-spotify
description: how to open Spotify on this machine
agent: apps
triggers: spotify, play music
---
Open Spotify with: powershell.exe -Command "Start-Process spotify:"
```

`curl -X POST localhost:8000/skills/reload` — then say *"Heyo, open spotify"*.

## Plugging in MCP servers

`mcp.json`:

```json
{
  "mcpServers": {
    "weather": {"command": "uvx", "args": ["some-weather-mcp"]},
    "search":  {"url": "http://localhost:9000/mcp"}
  }
}
```

Restart the server; an `mcp` agent appears in the graph with every tool the servers
expose, and the router learns to dispatch to it.

## Configuration

| file | role |
|---|---|
| `.env` (from `.env.example`) | ports, URLs, workspace dir |
| `models.yaml` (generated) | role → backend+model mapping |
| `mcp.json` | your MCP servers |
| `skills/*.md` | taught behaviors |

vLLM instead of / alongside Ollama (needs ≥16GB VRAM for useful models):

```bash
docker compose --profile vllm up -d vllm      # then point a role's backend to "vllm"
```

## Development

```bash
uv run pytest          # mocked-LLM test suite (routing, agents, sandbox, skills, SSE)
uv run ruff check .
docker compose --profile full up --build      # run the API in Docker too
```

### Repo map

```
heyo/        server: config, llm client, graph (router/agents), memory, skills, api, voice_io
apps/voice/  thin voice client (wake word + hotkey + mic/speaker) + Windows setup.ps1
ui/          single-file web console (no build step)
skills/      taught .md skills        scripts/   setup + model-fit + wake-word bench
```

## Hardware notes

Developed against a 6GB GTX 1660 Ti: `qwen2.5-coder:3b` for routing/agents/chat,
`nomic-embed-text` embeddings, faster-whisper `small` (int8) — all comfortably
resident. `fit_models.py` re-derives this for whatever hardware it finds.

Hard-won performance lessons baked into the defaults (measured on the 1660 Ti):

- **Reasoning models: stream the thinking, don't hide it.** qwen3's `<think>`
  blocks cost seconds per step — but 3B non-reasoning models hallucinate on
  research/tool tasks. Heyo streams reasoning live into the UI ("thinking" SSE
  events), so qwen3 is the default for agents; the router uses `/no_think`
  (reliable single-turn) plus deterministic fast-paths. Swap `general` to
  `qwen2.5-coder:3b` in models.yaml if you prefer speed over smarts.
- **Avoid Ollama's `response_format` (grammar-constrained JSON)** on consumer GPUs:
  measured 42s vs 4s for the same routing call with prompt-based JSON + parsing.
- **Keep prompts short.** Prefill ran at ~266 tok/s (no tensor cores) while
  generation hit ~59 tok/s — on cards like this, prompt length dominates latency.
  Skills are injected only into the agent they target; history windows are tight.
- **Small models botch tool-call formatting** — the agent loop parses tool calls
  written as JSON text (soft tool calls) instead of failing.

Measured end-to-end after tuning: chat first token ~1.3s, file operation ~2.2s,
app launch ~7s (was 19–56s before).
