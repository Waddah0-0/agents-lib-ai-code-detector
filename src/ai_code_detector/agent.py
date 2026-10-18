"""The AI Code Detector agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agents_core import AgentResult, BaseAgent, RunContext
from agents_core.llm import LLMClient

from ai_code_detector.markers import DEFAULT_MARKERS, load_markers
from ai_code_detector.methodology import DEFAULT_METHODOLOGY, load_methodology
from ai_code_detector.models import DetectorInput, DetectorOutput
from ai_code_detector.pipeline import detect


class AICodeDetector(BaseAgent):
    id = "ai-code-detector"
    name = "AI Code Detector"
    description = (
        "Estimates how likely a repository was written with AI assistance, as a range with a "
        "confidence level and the evidence for and against."
    )
    version = "0.1.0"
    InputModel = DetectorInput
    OutputModel = DetectorOutput
    primary_input = "target"

    def __init__(self, llm: LLMClient | None = None, config: Mapping[str, Any] | None = None):
        super().__init__(llm=llm, config=config)
        self.methodology = load_methodology(self.config.get("methodology", DEFAULT_METHODOLOGY))
        self.markers = load_markers(self.config.get("markers", DEFAULT_MARKERS))

    @property
    def methodology_version(self) -> str:
        return (
            f"detector-{self.version}+methodology-{self.methodology.version}"
            f"+markers-{self.markers.version}"
        )

    async def run(self, input: DetectorInput, ctx: RunContext) -> AgentResult:
        async with await ctx.load_workspace(
            input.target, paths=input.paths, branch=input.branch
        ) as ws:
            detection = await detect(
                ws, input, ctx, self.methodology, self.markers, self.methodology_version
            )
        ctx.report_progress("done", "Finished", 1.0)
        return self.make_result(
            summary=detection.summary,
            output=detection.output,
            findings=detection.findings,
            warnings=detection.warnings,
            status="partial" if detection.partial else "ok",
        )
