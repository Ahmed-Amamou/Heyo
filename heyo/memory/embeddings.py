"""Text embeddings via Ollama's OpenAI-compatible /v1/embeddings endpoint."""

from __future__ import annotations

import httpx

from heyo.config import ModelsConfig


class Embedder:
    def __init__(self, models: ModelsConfig):
        self.models = models
        self._http = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.models.base_url('embedder')}/embeddings"
        resp = await self._http.post(
            url,
            json={"model": self.models.role("embedder").model, "input": texts},
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
