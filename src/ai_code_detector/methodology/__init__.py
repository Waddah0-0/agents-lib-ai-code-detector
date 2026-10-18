"""Versioned weights and thresholds. Code reads numbers from here, never hard-codes them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Any

DEFAULT_METHODOLOGY = "v0"


@dataclass(frozen=True)
class Methodology:
    version: str
    data: dict[str, Any]

    @property
    def secondary_factor(self) -> float:
        return float(self.data["secondary_factor"])

    @property
    def min_code_lines(self) -> int:
        return int(self.data["min_code_lines"])

    @property
    def bias(self) -> float:
        return float(self.data["bias"])

    @property
    def file_weights(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.data["file_weights"].items()}

    @property
    def repo_weights(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.data["repo_weights"].items()}

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data[name])

    def thresholds_for(self, signal_id: str) -> dict[str, float]:
        return {k: float(v) for k, v in self.data["thresholds"][signal_id].items()}


def load_methodology(version: str = DEFAULT_METHODOLOGY) -> Methodology:
    text = (
        resources.files("ai_code_detector.methodology")
        .joinpath(f"{version}.toml")
        .read_text(encoding="utf-8")
    )
    data = tomllib.loads(text)
    return Methodology(version=str(data["version"]), data=data)
