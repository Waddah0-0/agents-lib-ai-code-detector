"""Detect explicit AI-tool markers: config files, co-author trailers, bot identities."""

from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources

from margin_core.loaders import GitCommit, Workspace

from ai_code_detector.models import MetadataMarker

DEFAULT_MARKERS = "v1"


@dataclass(frozen=True)
class MarkerSet:
    version: str
    config_files: tuple[tuple[str, str], ...]
    trailers: tuple[tuple[re.Pattern[str], str], ...]
    message_lines: tuple[tuple[re.Pattern[str], str], ...]
    identities: tuple[tuple[re.Pattern[str], str], ...]


def load_markers(version: str = DEFAULT_MARKERS) -> MarkerSet:
    text = (
        resources.files("ai_code_detector.markers")
        .joinpath(f"{version}.toml")
        .read_text(encoding="utf-8")
    )
    data = tomllib.loads(text)

    def compiled(key: str) -> tuple[tuple[re.Pattern[str], str], ...]:
        return tuple(
            (re.compile(entry["pattern"], re.IGNORECASE), entry["tool"])
            for entry in data.get(key, [])
        )

    return MarkerSet(
        version=str(data["version"]),
        config_files=tuple((e["path"], e["tool"]) for e in data.get("config_file", [])),
        trailers=compiled("trailer"),
        message_lines=compiled("message_line"),
        identities=compiled("identity"),
    )


def detect_config_files(ws: Workspace, markers: MarkerSet) -> list[MetadataMarker]:
    return [
        MetadataMarker(
            kind="config_file",
            tool=tool,
            location=path,
            detail=f"`{path}` is a configuration or instructions file for {tool}.",
        )
        for path, tool in markers.config_files
        if ws.exists(path)
    ]


def commit_marker_tools(commit: GitCommit, markers: MarkerSet) -> list[tuple[str, str]]:
    """``(kind, tool)`` pairs for every marker found in one commit."""
    found: list[tuple[str, str]] = []
    lines = [line.strip() for line in commit.message.splitlines()]
    for line in lines:
        if line.lower().startswith("co-authored-by:"):
            found += [("commit_trailer", tool) for pattern, tool in markers.trailers
                      if pattern.search(line)]  # fmt: skip
        found += [("generated_with_line", tool) for pattern, tool in markers.message_lines
                  if pattern.search(line)]  # fmt: skip
    identities = (
        f"{commit.author_name} <{commit.author_email}>",
        f"{commit.committer_name} <{commit.committer_email}>",
    )
    for pattern, tool in markers.identities:
        if any(pattern.search(identity) for identity in identities):
            found.append(("bot_author", tool))
    return list(dict.fromkeys(found))


def detect_commit_markers(
    commits: list[GitCommit], markers: MarkerSet
) -> tuple[list[MetadataMarker], set[str]]:
    """One marker per (kind, tool) with counts, plus the SHAs of every marked commit."""
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    marked: set[str] = set()
    for commit in commits:
        for key in commit_marker_tools(commit, markers):
            by_key[key].append(commit.sha)
            marked.add(commit.sha)
    labels = {
        "commit_trailer": "carry a co-author trailer for",
        "generated_with_line": "say they were generated with",
        "bot_author": "were authored or committed by",
    }
    result = []
    for (kind, tool), shas in by_key.items():
        result.append(
            MetadataMarker(
                kind=kind,  # type: ignore[arg-type]
                tool=tool,
                location=shas[0],
                detail=f"{len(shas)} of {len(commits)} commits {labels[kind]} {tool}.",
            )
        )
    return result, marked
