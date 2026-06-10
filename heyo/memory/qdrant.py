"""Qdrant-backed long-term memory and skill index."""

from __future__ import annotations

import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models as qm

from heyo.memory.embeddings import Embedder

MEMORY_COLLECTION = "heyo_memory"
SKILLS_COLLECTION = "heyo_skills"


class MemoryStore:
    def __init__(self, url: str, embedder: Embedder):
        self.client = AsyncQdrantClient(url=url)
        self.embedder = embedder
        self._dim: int | None = None

    async def close(self) -> None:
        await self.client.close()

    async def _ensure_collection(self, name: str) -> None:
        if self._dim is None:
            self._dim = len(await self.embedder.embed_one("dimension probe"))
        if not await self.client.collection_exists(name):
            await self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    # --- conversation memory ---

    async def remember(self, session_id: str, text: str) -> None:
        await self._ensure_collection(MEMORY_COLLECTION)
        vector = await self.embedder.embed_one(text)
        await self.client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": text, "session_id": session_id, "ts": time.time()},
                )
            ],
        )

    async def recall(self, query: str, limit: int = 4, min_score: float = 0.45) -> list[str]:
        await self._ensure_collection(MEMORY_COLLECTION)
        vector = await self.embedder.embed_one(query)
        hits = await self.client.query_points(
            collection_name=MEMORY_COLLECTION, query=vector, limit=limit, score_threshold=min_score
        )
        return [p.payload["text"] for p in hits.points]

    # --- skills index ---

    async def index_skills(self, skills: list[dict[str, Any]]) -> int:
        """Replace the skill index with the given skills (id is derived from name)."""
        await self._ensure_collection(SKILLS_COLLECTION)
        if not skills:
            return 0
        vectors = await self.embedder.embed(
            [f"{s['name']}: {s['description']}\n{s.get('triggers', '')}" for s in skills]
        )
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"heyo-skill:{s['name']}")),
                vector=v,
                payload=s,
            )
            for s, v in zip(skills, vectors)
        ]
        await self.client.delete_collection(SKILLS_COLLECTION)
        await self._ensure_collection(SKILLS_COLLECTION)
        await self.client.upsert(collection_name=SKILLS_COLLECTION, points=points)
        return len(points)

    async def find_skills(
        self, query: str, limit: int = 3, min_score: float = 0.4
    ) -> list[dict[str, Any]]:
        await self._ensure_collection(SKILLS_COLLECTION)
        vector = await self.embedder.embed_one(query)
        hits = await self.client.query_points(
            collection_name=SKILLS_COLLECTION, query=vector, limit=limit, score_threshold=min_score
        )
        return [p.payload for p in hits.points]
