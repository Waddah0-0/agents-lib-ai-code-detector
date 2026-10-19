"""Input and output models of the AI Code Detector."""

from __future__ import annotations

from typing import Literal

from agents_core import AgentInput, Evidence
from pydantic import BaseModel, Field, field_validator

Confidence = Literal["low", "medium", "high"]


class DetectorInput(AgentInput):
    target: str = Field(
        description="GitHub URL (https://github.com/<owner>/<repo>) or local path.",
        examples=["https://github.com/psf/requests"],
    )
    paths: list[str] | None = Field(
        default=None, description="Restrict the analysis to these files or folders."
    )
    branch: str | None = Field(default=None, description="Branch to analyse (GitHub only).")
    use_llm_judge: bool | None = Field(
        default=None,
        description=(
            "Ask a model to judge a sample of files. Default: when a model is configured. "
            "Without it the estimate rests on weak signals and confidence is low."
        ),
    )
    max_files_for_judge: int = Field(default=20, ge=1, le=200)
    max_judge_seconds: float = Field(default=300.0, gt=0)
    max_cost_usd: float | None = Field(
        default=None, ge=0, description="Judge spending limit; only matters for hosted models."
    )
    include_tests: bool = Field(
        default=False, description="Score test files too (reported separately)."
    )


class SignalHit(BaseModel):
    """One signal that fired, with its strength and the evidence behind it."""

    signal_id: str
    signal_version: str
    family: str
    strength: float = Field(ge=-1.0, le=1.0, description="Negative means evidence of a human.")
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)


class MetadataMarker(BaseModel):
    """An explicit trace of an AI tool: proves the tool was used, not how much."""

    kind: Literal["config_file", "commit_trailer", "bot_author", "generated_with_line"]
    tool: str
    location: str
    detail: str


MAX_QUOTE_CHARS = 160
MAX_WHY_CHARS = 100
MAX_EVIDENCE_ITEMS = 3
MAX_NOTES_CHARS = 160


def _trim(value: object, limit: int) -> object:
    return value[:limit] if isinstance(value, str) else value


class QuotedEvidence(BaseModel):
    """One quoted line. Limits keep local-model answers short (generation is the slow part);
    they are advertised in the JSON schema for constrained decoding and enforced by trimming,
    so an over-long answer is shortened rather than rejected."""

    line_start: int
    line_end: int
    quote: str = Field(json_schema_extra={"maxLength": MAX_QUOTE_CHARS})
    why: str = Field(json_schema_extra={"maxLength": MAX_WHY_CHARS})

    @field_validator("quote", mode="before")
    @classmethod
    def _short_quote(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.split("\n", 1)[0]  # one line only
        return _trim(value, MAX_QUOTE_CHARS)

    @field_validator("why", mode="before")
    @classmethod
    def _short_why(cls, value: object) -> object:
        return _trim(value, MAX_WHY_CHARS)


class JudgeVerdict(BaseModel):
    """A model's verdict on one file (phase 4)."""

    likelihood_ai: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_for: list[QuotedEvidence] = Field(
        default_factory=list, json_schema_extra={"maxItems": MAX_EVIDENCE_ITEMS}
    )
    evidence_against: list[QuotedEvidence] = Field(
        default_factory=list, json_schema_extra={"maxItems": MAX_EVIDENCE_ITEMS}
    )
    uncertainty_notes: str = Field(default="", json_schema_extra={"maxLength": MAX_NOTES_CHARS})

    @field_validator("evidence_for", "evidence_against", mode="before")
    @classmethod
    def _few_items(cls, value: object) -> object:
        return value[:MAX_EVIDENCE_ITEMS] if isinstance(value, list) else value

    @field_validator("uncertainty_notes", mode="before")
    @classmethod
    def _short_notes(cls, value: object) -> object:
        return _trim(value, MAX_NOTES_CHARS)


class FileScore(BaseModel):
    path: str
    language: str | None
    lines: int
    is_test: bool = False
    probability: float = Field(ge=0.0, le=1.0)
    signals: list[SignalHit] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    ai_commit_touches: int = Field(
        default=0, description="Commits with AI markers that changed this file."
    )


class DetectorOutput(BaseModel):
    methodology_version: str
    estimate: float = Field(ge=0.0, le=1.0)
    estimate_range: tuple[float, float]
    confidence: Confidence
    confidence_reasons: list[str] = Field(default_factory=list)
    metadata_markers: list[MetadataMarker] = Field(default_factory=list)
    repo_signals: list[SignalHit] = Field(default_factory=list)
    files: list[FileScore] = Field(default_factory=list)
    files_analyzed: int
    files_skipped: int
    languages: dict[str, int] = Field(default_factory=dict)
    unsupported_languages: dict[str, int] = Field(default_factory=dict)
