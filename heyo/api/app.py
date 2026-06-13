"""FastAPI application factory and `heyo-api` entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from heyo import __version__
from heyo.api.chat import router as chat_router
from heyo.api.sessions import SessionStore
from heyo.api.voice import router as voice_router
from heyo.config import get_settings
from heyo.graph.agents import apps as apps_agent
from heyo.graph.agents import mcp as mcp_agent
from heyo.graph.agents import web as web_agent
from heyo.graph.build import build_graph
from heyo.llm.client import LLMClient
from heyo.mcp.manager import MCPManager, load_mcp_config
from heyo.memory.embeddings import Embedder
from heyo.memory.qdrant import MemoryStore
from heyo.skills.loader import load_skills

log = logging.getLogger("heyo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    models = settings.load_models()
    llm = LLMClient(models)
    embedder = Embedder(models)
    memory = MemoryStore(settings.qdrant_url, embedder)

    browser = web_agent.Browser()
    mcp = MCPManager(load_mcp_config(settings.heyo_mcp_file))
    try:
        await mcp.start()
        if mcp.has_servers:
            log.info("MCP tools: %s", mcp.tool_names())
    except Exception as exc:
        log.warning("MCP servers failed to start: %s", exc)

    extra_agents = {
        "apps": (apps_agent.make_apps_agent(llm), apps_agent.DESCRIPTION),
        "web": (web_agent.make_web_agent(llm, browser), web_agent.DESCRIPTION),
    }
    if mcp.tool_names():
        extra_agents["mcp"] = (mcp_agent.make_mcp_agent(llm, mcp), mcp_agent.description(mcp))

    app.state.settings = settings
    app.state.llm = llm
    app.state.embedder = embedder
    app.state.memory = memory
    app.state.mcp = mcp
    app.state.sessions = SessionStore()
    app.state.graph = build_graph(llm, settings, memory=memory, extra_agents=extra_agents)

    try:
        count = await memory.index_skills(load_skills(settings.heyo_skills_dir))
        log.info("indexed %d skills", count)
    except Exception as exc:
        log.warning("skill indexing skipped (qdrant/ollama unavailable?): %s", exc)

    yield
    await browser.close()
    await mcp.stop()
    await memory.close()
    await embedder.close()
    await llm.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Heyo", version=__version__, lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(voice_router)

    @app.get("/health")
    async def health():
        models = app.state.settings.load_models()
        return {
            "status": "ok",
            "version": __version__,
            "roles": {name: rc.model for name, rc in models.roles.items()},
        }

    @app.post("/skills/reload")
    async def reload_skills():
        skills = load_skills(app.state.settings.heyo_skills_dir)
        count = await app.state.memory.index_skills(skills)
        return {"indexed": count, "skills": [s["name"] for s in skills]}

    ui_index = Path(__file__).resolve().parent.parent.parent / "ui" / "index.html"
    if ui_index.exists():

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(ui_index)

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "heyo.api.app:create_app",
        factory=True,
        host=settings.heyo_host,
        port=settings.heyo_port,
    )


if __name__ == "__main__":
    main()
