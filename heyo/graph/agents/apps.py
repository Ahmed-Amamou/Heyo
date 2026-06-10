"""App automation agent: opens Windows applications from WSL via interop.

The agent only knows HOW to launch things; WHAT to launch for a given request
is taught through skill .md files (see skills/_examples/open-app.md), which the
prepare node injects as skill_context.
"""

from __future__ import annotations

import asyncio
import shlex

from heyo.graph.agents.base import ToolKit, make_tool_agent
from heyo.llm.client import LLMClient

DESCRIPTION = "open or launch Windows applications, URLs, and folders on the user's machine"

APPS_PROMPT = """\
You are Heyo's app-automation agent running inside WSL2. You launch Windows
applications with your run_command tool using Windows interop binaries:
`cmd.exe /c start <app|url|path>`, `powershell.exe -Command "Start-Process ..."`,
or `explorer.exe <path>`.

Follow any taught skills exactly — they contain the user's preferred commands.
Only run launch/open commands; refuse anything destructive. Confirm in one short
sentence what you launched.
"""

ALLOWED_BINARIES = {"cmd.exe", "powershell.exe", "explorer.exe", "wslview"}


def make_apps_toolkit(timeout: float = 15.0) -> ToolKit:
    kit = ToolKit()

    @kit.add(
        "run_command",
        "Run a Windows-interop launch command (cmd.exe /c start ..., powershell.exe ..., "
        "explorer.exe ...)",
        {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "full command line"}},
            "required": ["command"],
        },
    )
    async def run_command(command: str) -> str:
        parts = shlex.split(command)
        if not parts or parts[0] not in ALLOWED_BINARIES:
            return (
                f"error: only these launchers are allowed: {sorted(ALLOWED_BINARIES)}. "
                "Rewrite the command to use one of them."
            )
        proc = await asyncio.create_subprocess_exec(
            *parts, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return "launched (still running)"
        text = out.decode(errors="replace").strip()
        return f"exit={proc.returncode}" + (f"\n{text[:1000]}" if text else "")

    return kit


def make_apps_agent(llm: LLMClient):
    return make_tool_agent("apps", llm, "general", APPS_PROMPT, make_apps_toolkit())
