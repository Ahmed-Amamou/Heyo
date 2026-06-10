from __future__ import annotations

from pathlib import Path

from heyo.skills.loader import format_skills, load_skills

EXAMPLE = """---
name: test-skill
description: a test skill
agent: apps
triggers: do the thing
---

Step 1: do the thing.
"""


def test_load_skills_parses_frontmatter(tmp_path: Path):
    (tmp_path / "test.md").write_text(EXAMPLE)
    skills = load_skills(tmp_path)
    assert len(skills) == 1
    s = skills[0]
    assert s["name"] == "test-skill"
    assert s["agent"] == "apps"
    assert "Step 1" in s["body"]


def test_load_skills_defaults_from_filename(tmp_path: Path):
    (tmp_path / "no-frontmatter.md").write_text("Just instructions.")
    skills = load_skills(tmp_path)
    assert skills[0]["name"] == "no-frontmatter"
    assert skills[0]["agent"] == "any"


def test_load_skills_missing_dir():
    assert load_skills(Path("/nonexistent")) == []


def test_format_skills():
    text = format_skills(
        [{"name": "a", "agent": "apps", "body": "body-a"},
         {"name": "b", "agent": "any", "body": "body-b"}]
    )
    assert "## Skill: a (agent: apps)" in text and "body-b" in text


def test_repo_example_skill_parses():
    skills = load_skills(Path(__file__).parent.parent / "skills")
    names = [s["name"] for s in skills]
    assert "open-windows-app" in names
