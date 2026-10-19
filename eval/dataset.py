"""Evaluation datasets: a JSONL manifest of labelled samples, fetched into a local cache.

Third-party code is never copied into this repository. A ``git`` sample names a public GitHub
repository and an exact commit; the fetcher clones it (blobless) into the cache and checks
that commit out, so history ends exactly there. A ``local`` sample points to a folder next to
the manifest (used for small self-generated samples and for tests).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal

from margin_core.errors import InvalidTargetError
from margin_core.loaders import parse_github_url
from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["human", "ai", "mixed"]


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["git", "local"]
    url: str | None = None  # git: https://github.com/<owner>/<repo>
    commit: str | None = None  # git: full 40-character SHA
    path: str | None = None  # local: folder relative to the manifest; git: sub-folder to analyse

    @model_validator(mode="after")
    def _check(self) -> Source:
        if self.kind == "git":
            if not self.url or not self.commit or len(self.commit) != 40:
                raise ValueError("git sources need a GitHub url and a full 40-character commit")
            try:
                parse_github_url(self.url)
            except InvalidTargetError as exc:
                raise ValueError(str(exc)) from exc
        elif not self.path:
            raise ValueError("local sources need a path")
        return self


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    label: Label
    ai_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Source
    language: str = "python"
    era: Literal["pre2021", "2021-2022", "post2023", "unknown"]
    tool: str | None = None  # AI tool that produced it, when known
    provenance: str  # how we know the label
    provenance_confidence: Literal["low", "medium", "high"]
    split: Literal["train", "val", "test"]

    @property
    def is_ai(self) -> bool:
        """Positive class: AI, or mixed with at least half AI-written code."""
        return self.label == "ai" or (self.label == "mixed" and (self.ai_fraction or 0) >= 0.5)


def load_manifest(path: Path) -> list[Sample]:
    samples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                samples.append(Sample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [s.id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample ids")
    return samples


def cache_dir() -> Path:
    base = os.environ.get("MARGIN_CACHE_DIR") or str(Path.home() / ".cache" / "margin")
    return Path(base) / "eval"


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", "-c", "advice.detachedHead=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        timeout=600,
    )


def fetch(sample: Sample, manifest_dir: Path, cache: Path | None = None) -> Path:
    """The folder to analyse for ``sample``, cloning it first if needed."""
    source = sample.source
    if source.kind == "local":
        assert source.path is not None
        return (manifest_dir / source.path).resolve()
    assert source.url is not None and source.commit is not None
    target = (cache or cache_dir()) / f"{sample.id}-{source.commit[:12]}"
    marker = target / ".margin-fetched"
    if not marker.exists():
        if target.exists():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "-q", "--filter=blob:none", "--no-checkout", "--", source.url, str(target))
        _git("checkout", "-q", "--detach", source.commit, cwd=target)
        marker.write_text(json.dumps({"url": source.url, "commit": source.commit}), "utf-8")
    return target / source.path if source.path else target
