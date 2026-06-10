#!/usr/bin/env bash
# Heyo setup: installs uv, Ollama (native, GPU), llmfit, and Python deps.
set -euo pipefail

info() { printf '\033[1;34m[heyo]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[heyo]\033[0m %s\n' "$*"; }

# --- uv ---
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  info "uv already installed."
fi

# --- Ollama (native install: best GPU support under WSL2) ---
if ! command -v ollama >/dev/null 2>&1; then
  info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  info "Ollama already installed."
fi

# --- llmfit (hardware fit checker) ---
if ! command -v llmfit >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    info "Installing llmfit via brew..."
    brew install llmfit
  else
    warn "llmfit not installed and brew unavailable."
    warn "Install manually: https://github.com/AlexsJones/llmfit (scripts/fit_models.py falls back to safe defaults)."
  fi
else
  info "llmfit already installed."
fi

# --- Docker check ---
if ! command -v docker >/dev/null 2>&1; then
  warn "Docker not available in this WSL distro."
  warn "Enable: Docker Desktop > Settings > Resources > WSL integration > this distro."
fi

# --- Python deps ---
info "Syncing Python dependencies..."
uv sync --extra dev

info "Done. Next: ensure 'ollama serve' is running, then: uv run python scripts/fit_models.py"
