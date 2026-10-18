"""Fixture repositories built with git inside the test, so no network or stored code is needed."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

GitRunner = Callable[..., str]
FIXTURES = Path(__file__).parent / "fixtures"


def run_git(
    repo: Path,
    *args: str,
    date: str = "2024-01-01T12:00:00+00:00",
    author: str = "Test User <test@example.com>",
) -> str:
    name, email = author.rsplit(" <", 1)
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email.rstrip(">"),
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email.rstrip(">"),
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout


AI_STYLE_FILE = (FIXTURES / "ai_style.py.txt").read_text(encoding="utf-8")

HUMAN_STYLE_FILE = (FIXTURES / "human_style.py.txt").read_text(encoding="utf-8")


@pytest.fixture
def ai_repo(tmp_path: Path) -> Path:
    """A project that arrived in one AI co-authored commit, with AI-style code and README."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "ai_repo"
    (repo / "src" / "app").mkdir(parents=True)
    run_git(repo, "init", "-q", "-b", "main")
    (repo / "CLAUDE.md").write_text("# Instructions for Claude\n", encoding="utf-8")
    (repo / "README.md").write_text(
        "# Project\n\n## 🚀 Features\n\nA powerful, seamless and robust tool.\n\n"
        "## 📦 Installation\n\n## 💡 Usage\n\n## 🤝 Contributing\n\n## 📄 License\n",
        encoding="utf-8",
    )
    for i in range(12):
        (repo / "src" / "app" / f"module_{i}.py").write_text(AI_STYLE_FILE, encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(
        repo,
        "commit",
        "-q",
        "-m",
        "Initial implementation\n\n🤖 Generated with Claude Code\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
    )
    return repo


@pytest.fixture
def human_repo(tmp_path: Path) -> Path:
    """A project grown over months with messy commits and human-style code."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "human_repo"
    (repo / "tool").mkdir(parents=True)
    run_git(repo, "init", "-q", "-b", "main")
    messages = ["first", "wip", "fix typo", "oops", "more stuff", "Revert last change",
                "try again", "parser works now mostly", "cleanup", "asdf", "tmp", "minor",
                "handle empty lines", "forgot file"]  # fmt: skip
    for i, message in enumerate(messages):
        target = repo / "tool" / f"t{i % 12}.py"
        target.write_text(HUMAN_STYLE_FILE + f"\n# rev {i}\n", encoding="utf-8")
        run_git(repo, "add", ".")
        month = 1 + i % 12
        run_git(
            repo,
            "commit",
            "-q",
            "-m",
            message,
            date=f"2019-{month:02d}-0{1 + i % 9}T10:00:00+00:00",
        )
    return repo


@pytest.fixture
def ai_code() -> str:
    return AI_STYLE_FILE


@pytest.fixture
def human_code() -> str:
    return HUMAN_STYLE_FILE
