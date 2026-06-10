#!/usr/bin/env python3
"""Hardware-fit model selection for Heyo.

Uses llmfit (https://github.com/AlexsJones/llmfit) when available to recommend
models that actually fit this machine, then pulls them with Ollama and writes
models.yaml mapping Heyo roles -> models.

Falls back to conservative 6GB-VRAM defaults if llmfit is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = REPO_ROOT / "models.yaml"

# Conservative defaults for ~6GB VRAM (GTX 1660 Ti class hardware).
# Non-reasoning model on purpose: reasoning models (qwen3 etc.) burn seconds of
# hidden <think> tokens per pipeline step, and their /no_think switch is
# unreliable in multi-turn tool conversations.
DEFAULTS = {
    "router": "qwen2.5-coder:3b",
    "general": "qwen2.5-coder:3b",
    "coder": "qwen2.5-coder:3b",
    "embedder": "nomic-embed-text",
}


def llmfit_recommendations() -> list[str]:
    """Return model names recommended by llmfit, best-first. Empty list on failure."""
    if not shutil.which("llmfit"):
        print("[fit] llmfit not found; using built-in defaults for 6GB VRAM.")
        return []
    for args in (["llmfit", "--json"], ["llmfit", "--output", "json"], ["llmfit"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=120).stdout
        except (subprocess.TimeoutExpired, OSError):
            continue
        models = _parse_llmfit_output(out)
        if models:
            print(f"[fit] llmfit recommends (best first): {models[:5]}")
            return models
    print("[fit] could not parse llmfit output; using defaults.")
    return []


def _parse_llmfit_output(out: str) -> list[str]:
    """Parse llmfit output (JSON preferred, plain text fallback)."""
    try:
        data = json.loads(out)
        items = data if isinstance(data, list) else data.get("models", data.get("results", []))
        names = [m.get("name") or m.get("model") for m in items if isinstance(m, dict)]
        return [n for n in names if n]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Plain-text: take lines that look like model ids (name:tag or org/name)
    names = []
    for line in out.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token and (":" in token or "/" in token) and not token.startswith(("http", "-")):
            names.append(token)
    return names


def ollama_pull(model: str) -> bool:
    print(f"[fit] pulling {model} ...")
    res = subprocess.run(["ollama", "pull", model])
    return res.returncode == 0


def main() -> int:
    if not shutil.which("ollama"):
        print("[fit] ERROR: ollama not installed. Run scripts/setup.sh first.", file=sys.stderr)
        return 1

    recommended = llmfit_recommendations()
    roles = dict(DEFAULTS)
    if recommended:
        # Use llmfit's top chat-capable pick for router/general when it is an Ollama-style id.
        top = next((m for m in recommended if "embed" not in m.lower() and "/" not in m), None)
        if top:
            roles["router"] = top
            roles["general"] = top

    print(f"[fit] role assignment: {roles}")
    failed = [m for m in dict.fromkeys(roles.values()) if not ollama_pull(m)]
    if failed:
        print(f"[fit] WARNING: failed to pull: {failed}", file=sys.stderr)

    config = {
        "backends": {
            "ollama": {"base_url": "http://localhost:11434/v1"},
            "vllm": {"base_url": "http://localhost:8001/v1"},
        },
        "roles": {role: {"backend": "ollama", "model": model} for role, model in roles.items()},
    }
    MODELS_FILE.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"[fit] wrote {MODELS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
