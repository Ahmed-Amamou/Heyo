"""Skill loader: parses skills/*.md (YAML frontmatter + body) — how the user teaches Heyo.

Frontmatter fields:
    name:        unique slug (defaults to filename)
    description: one line, used for retrieval matching
    agent:       which agent this skill applies to (chat|files|web|apps|mcp|any)
    triggers:    optional comma-separated phrases that should activate the skill
The markdown body is the instruction text injected into the agent's system prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter


def load_skills(skills_dir: Path) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    if not skills_dir.is_dir():
        return skills
    for path in sorted(skills_dir.rglob("*.md")):
        post = frontmatter.load(path)
        name = str(post.get("name") or path.stem)
        skills.append(
            {
                "name": name,
                "description": str(post.get("description") or name),
                "agent": str(post.get("agent") or "any"),
                "triggers": str(post.get("triggers") or ""),
                "body": post.content.strip(),
                "path": str(path),
            }
        )
    return skills


def format_skills(skills: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"## Skill: {s['name']} (agent: {s['agent']})\n{s['body']}" for s in skills
    )
