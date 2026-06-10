"""Heyo configuration: env vars + models.yaml role mapping."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoleConfig(BaseModel):
    backend: str = "ollama"
    model: str


class ModelsConfig(BaseModel):
    backends: dict[str, dict[str, str]] = {
        "ollama": {"base_url": "http://localhost:11434/v1"},
        "vllm": {"base_url": "http://localhost:8001/v1"},
    }
    roles: dict[str, RoleConfig] = {}

    def role(self, name: str) -> RoleConfig:
        if name in self.roles:
            return self.roles[name]
        if "general" in self.roles:
            return self.roles["general"]
        raise KeyError(f"no model configured for role '{name}' and no 'general' fallback")

    def base_url(self, role_name: str) -> str:
        return self.backends[self.role(role_name).backend]["base_url"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    heyo_host: str = "0.0.0.0"
    heyo_port: int = 8000
    heyo_workspace: Path = Path("./workspace")
    heyo_models_file: Path = Path("./models.yaml")
    heyo_mcp_file: Path = Path("./mcp.json")
    heyo_skills_dir: Path = Path("./skills")
    ollama_base_url: str = "http://localhost:11434/v1"
    vllm_base_url: str = "http://localhost:8001/v1"
    qdrant_url: str = "http://localhost:6333"

    def load_models(self) -> ModelsConfig:
        if self.heyo_models_file.exists():
            data = yaml.safe_load(self.heyo_models_file.read_text()) or {}
            cfg = ModelsConfig.model_validate(data)
        else:
            # Defaults for first boot before fit_models.py has run.
            cfg = ModelsConfig(
                roles={
                    "router": RoleConfig(model="qwen3:4b"),
                    "general": RoleConfig(model="qwen3:4b"),
                    "embedder": RoleConfig(model="nomic-embed-text"),
                }
            )
        # Env vars override base URLs from the file (used in docker).
        cfg.backends.setdefault("ollama", {})["base_url"] = self.ollama_base_url
        cfg.backends.setdefault("vllm", {})["base_url"] = self.vllm_base_url
        return cfg


@lru_cache
def get_settings() -> Settings:
    return Settings()
