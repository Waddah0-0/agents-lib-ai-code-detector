"""Marker detection, using the exact formats verified against real public commits."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from agents_core.loaders import GitCommit, LocalLoader
from ai_code_detector.markers import (
    commit_marker_tools,
    detect_commit_markers,
    detect_config_files,
    load_markers,
)

MARKERS = load_markers()


def commit(
    message: str = "Change things",
    author: str = "Dev <dev@example.com>",
    committer: str = "GitHub <noreply@github.com>",
    sha: str = "a" * 40,
) -> GitCommit:
    an, ae = author[:-1].split(" <")
    cn, ce = committer[:-1].split(" <")
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return GitCommit(
        sha=sha, author_name=an, author_email=ae, authored_at=now, committer_name=cn,
        committer_email=ce, committed_at=now, parents=[], message=message,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"message": "x\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"},
         ("commit_trailer", "Claude")),
        ({"message": "x\n\nCo-Authored-By: Gemini <noreply@google.com>"},
         ("commit_trailer", "Gemini")),
        ({"message": "x\n\nCo-authored-by: Codex <noreply@openai.com>"},
         ("commit_trailer", "OpenAI Codex")),
        ({"message": "x\n\nCo-authored-by: Cursor <cursoragent@cursor.com>"},
         ("commit_trailer", "Cursor")),
        ({"message": "x\n\n🤖 Generated with Claude Code"},
         ("generated_with_line", "Claude Code")),
        ({"message": "x\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"},
         ("generated_with_line", "Claude Code")),
        ({"author": "“aider” (aider) <aider@example.com>"}, ("bot_author", "Aider")),
        ({"author": "copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>"},
         ("bot_author", "GitHub Copilot coding agent")),
        ({"author": "Devin AI <158243242+devin-ai-integration[bot]@users.noreply.github.com>"},
         ("bot_author", "Devin")),
        ({"committer": "google-labs-jules[bot] "
                       "<161369871+google-labs-jules[bot]@users.noreply.github.com>"},
         ("bot_author", "Jules")),
    ],
)  # fmt: skip
def test_verified_marker_formats(kwargs: dict[str, str], expected: tuple[str, str]) -> None:
    assert commit_marker_tools(commit(**kwargs), MARKERS) == [expected]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"message": "Fix parser\n\nCo-authored-by: Jane <jane@example.com>"},
        {"message": "Mention noreply@anthropic.com in docs"},  # not a trailer line
        {"author": "Aiden Smith <aider@example.com>"},
        {"message": "Generated with care by hand"},
    ],
)
def test_human_commits_have_no_markers(kwargs: dict[str, str]) -> None:
    assert commit_marker_tools(commit(**kwargs), MARKERS) == []


def test_commit_markers_are_counted_per_tool() -> None:
    trailer = "x\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    commits = [
        commit(trailer, sha="1" * 40),
        commit("plain", sha="2" * 40),
        commit(trailer, sha="3" * 40),
    ]
    markers, marked = detect_commit_markers(commits, MARKERS)
    assert len(markers) == 1
    assert markers[0].detail == "2 of 3 commits carry a co-author trailer for Claude."
    assert marked == {"1" * 40, "3" * 40}


async def test_config_files_including_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / ".cursor" / "rules").mkdir(parents=True)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "copilot-instructions.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("x", encoding="utf-8")
    ws = await LocalLoader().load(str(tmp_path))
    tools = sorted(m.tool for m in detect_config_files(ws, MARKERS))
    assert tools == ["Claude Code", "Cursor", "GitHub Copilot"]


def test_every_commit_marker_records_its_verified_example() -> None:
    import tomllib
    from importlib import resources

    data = tomllib.loads(
        resources.files("ai_code_detector.markers").joinpath("v1.toml").read_text("utf-8")
    )
    for key in ("trailer", "message_line", "identity"):
        for entry in data[key]:
            assert "@" in entry.get("example", ""), entry
