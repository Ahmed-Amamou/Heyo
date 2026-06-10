"""File management agent, sandboxed to the HEYO_WORKSPACE directory."""

from __future__ import annotations

from pathlib import Path

from heyo.graph.agents.base import ToolKit, make_tool_agent
from heyo.llm.client import LLMClient

DESCRIPTION = "create, read, list, move or delete files and folders in the user's workspace"

FILES_PROMPT = """\
You are Heyo's file-management agent. You operate ONLY inside the user's workspace
directory using your tools. Paths are relative to the workspace root. After acting,
summarize what you did in one or two sentences.
"""

_PATH_PARAM = {"path": {"type": "string", "description": "path relative to workspace root"}}


def make_files_toolkit(workspace: Path) -> ToolKit:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    kit = ToolKit()

    def safe(rel: str) -> Path:
        p = (workspace / rel).resolve()
        if not p.is_relative_to(workspace):
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    @kit.add(
        "list_dir",
        "List files and directories at a path",
        {"type": "object", "properties": _PATH_PARAM, "required": []},
    )
    def list_dir(path: str = ".") -> str:
        target = safe(path)
        if not target.is_dir():
            return f"not a directory: {path}"
        entries = sorted(target.iterdir())
        if not entries:
            return "(empty)"
        return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.relative_to(workspace)}" for e in entries)

    @kit.add(
        "read_file",
        "Read a text file's contents",
        {"type": "object", "properties": _PATH_PARAM, "required": ["path"]},
    )
    def read_file(path: str) -> str:
        return safe(path).read_text()[:20000]

    @kit.add(
        "write_file",
        "Create or overwrite a text file",
        {
            "type": "object",
            "properties": {**_PATH_PARAM, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )
    def write_file(path: str, content: str) -> str:
        target = safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @kit.add(
        "move",
        "Move or rename a file or directory",
        {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dest": {"type": "string"}},
            "required": ["src", "dest"],
        },
    )
    def move(src: str, dest: str) -> str:
        s, d = safe(src), safe(dest)
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
        return f"moved {src} -> {dest}"

    @kit.add(
        "delete",
        "Delete a file or an empty directory",
        {"type": "object", "properties": _PATH_PARAM, "required": ["path"]},
    )
    def delete(path: str) -> str:
        target = safe(path)
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()
        return f"deleted {path}"

    @kit.add(
        "mkdir",
        "Create a directory (with parents)",
        {"type": "object", "properties": _PATH_PARAM, "required": ["path"]},
    )
    def mkdir(path: str) -> str:
        safe(path).mkdir(parents=True, exist_ok=True)
        return f"created {path}"

    return kit


def make_files_agent(llm: LLMClient, workspace: Path):
    return make_tool_agent("files", llm, "general", FILES_PROMPT, make_files_toolkit(workspace))
