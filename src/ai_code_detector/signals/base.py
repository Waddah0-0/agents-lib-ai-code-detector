"""Signal interfaces and the context they read from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from agents_core import Evidence
from agents_core.loaders import GitCommit, Workspace

from ai_code_detector.methodology import Methodology
from ai_code_detector.models import SignalHit
from ai_code_detector.python_source import PythonFile


@dataclass
class RepoContext:
    workspace: Workspace
    methodology: Methodology
    commits: list[GitCommit] = field(default_factory=list)  # newest first, scoped to the target
    changed_files: dict[str, list[str]] = field(default_factory=dict)  # sha -> workspace paths
    history_truncated: bool = False


class _Signal(ABC):
    id: ClassVar[str]
    version: ClassVar[str]
    family: ClassVar[str]

    def hit(self, strength: float, explanation: str, evidence: list[Evidence]) -> SignalHit:
        return SignalHit(
            signal_id=self.id,
            signal_version=self.version,
            family=self.family,
            strength=max(-1.0, min(1.0, round(strength, 4))),
            explanation=explanation,
            evidence=evidence,
        )


class RepoSignal(_Signal):
    """Looks at the repository as a whole (history, README)."""

    @abstractmethod
    def evaluate(self, ctx: RepoContext) -> list[SignalHit]: ...


class FileSignal(_Signal):
    """Looks at one parsed Python file."""

    @abstractmethod
    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]: ...


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def ramp(value: float, low: float, high: float) -> float:
    """0 at or below ``low``, 1 at or above ``high``, linear in between."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clamp01((value - low) / (high - low))
