"""Ask a model to judge files, with injection defences, quote checks, caching and budgets.

Design (PLAN §9.5):

- The code goes between random per-run markers and the prompt says it is data.
- The model never sees the heuristic signal scores.
- Long files are cut to one window of at most ``window_lines`` lines, chosen to contain the
  most lines the heuristics flagged; the prompt stays within ``token_budget`` tokens.
- The model points at line numbers; the quoted text is taken from the file, so evidence
  cannot be invented. Pointers outside the window are dropped, and a verdict with no
  valid pointer on either side is discarded.
- Verdicts are cached on disk by (provider, model, prompt version, window content).
- A share of files is judged a second time with the rubric in a different order; large
  disagreement lowers the run's confidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from margin_core.errors import BudgetExceededError, LLMError
from margin_core.llm import LLMClient
from pydantic import BaseModel, Field, field_validator

from ai_code_detector.models import (
    MAX_EVIDENCE_ITEMS,
    MAX_NOTES_CHARS,
    MAX_WHY_CHARS,
    FileScore,
    JudgeVerdict,
    QuotedEvidence,
)

PROMPT_VERSION = "judge_v2"  # v2: the model points at line numbers; we fill in the quotes


class LinePointer(BaseModel):
    """What the model returns per piece of evidence: a line number and a short reason."""

    line: int
    why: str = Field(json_schema_extra={"maxLength": MAX_WHY_CHARS})

    @field_validator("why", mode="before")
    @classmethod
    def _short(cls, value: object) -> object:
        return value[:MAX_WHY_CHARS] if isinstance(value, str) else value


MAX_FAILURES_IN_A_ROW = 3  # a model that cannot load fails every call; stop early


def short_error(exc: Exception) -> str:
    """The useful part of an LLM error: the server's own message when it sent JSON."""
    text = str(exc).strip()
    match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    message = match.group(1) if match else text
    if "requires more system memory" in message:
        message += " (close other programs, or use a smaller model)"
    return message


class JudgeAnswer(BaseModel):
    """The model's raw answer. Pointing at lines instead of quoting keeps answers short
    (Ollama does not enforce maxLength, and qwen3.5 quoted a 97-line docstring) and makes
    invented quotes impossible."""

    likelihood_ai: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_for: list[LinePointer] = Field(default_factory=list)
    evidence_against: list[LinePointer] = Field(default_factory=list)
    uncertainty_notes: str = ""

    @field_validator("evidence_for", "evidence_against", mode="before")
    @classmethod
    def _few(cls, value: object) -> object:
        return value[:MAX_EVIDENCE_ITEMS] if isinstance(value, list) else value

    @field_validator("uncertainty_notes", mode="before")
    @classmethod
    def _short_notes(cls, value: object) -> object:
        return value[:MAX_NOTES_CHARS] if isinstance(value, str) else value


SCHEMA_TEXT = json.dumps(JudgeAnswer.model_json_schema())
CHARS_PER_TOKEN = 3  # code is denser than prose (measured ~3 chars/token with qwen2.5)
RUBRIC = (
    "Comments that restate the next line of code, or explain the obvious.",
    "A docstring or doc comment on every function, including trivial ones.",
    "Uniform, textbook structure with no personal habits or shortcuts.",
    "Generic, overly descriptive names (process_data, result_list, handle_request).",
    "Defensive boilerplate: blanket error handling, redundant validation, excess logging.",
    "Human traces: TODOs with names or dates, commented-out code, debug prints, odd hacks.",
    "Inconsistencies: mixed styles, abandoned approaches, uneven care across the file.",
    "Domain shorthand or jargon that only someone close to the project would use.",
)
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class JudgeSettings:
    window_lines: int = 250
    token_budget: int = 6000
    max_output_tokens: int = 700
    agreement_share: float = 0.2
    agreement_min_files: int = 2
    disagreement_threshold: float = 0.3


@dataclass
class JudgeRun:
    verdicts: dict[str, JudgeVerdict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    partial: bool = False
    disagreement: float | None = None  # mean |difference| on re-judged files
    cache_hits: int = 0


def load_prompt(version: str = PROMPT_VERSION) -> str:
    return (
        resources.files("ai_code_detector.judge.prompts")
        .joinpath(f"{version}.md")
        .read_text(encoding="utf-8")
    )


def build_system_prompt(open_marker: str, close_marker: str, rubric_seed: int | None) -> str:
    items = list(RUBRIC)
    if rubric_seed is not None:
        random.Random(rubric_seed).shuffle(items)
    rubric = "\n".join(f"- {item}" for item in items)
    return load_prompt().format(open=open_marker, close=close_marker, rubric=rubric)


def choose_window(lines: Sequence[str], flagged: Sequence[int], max_lines: int) -> tuple[int, int]:
    """1-based inclusive line range of at most ``max_lines`` covering most flagged lines."""
    if len(lines) <= max_lines:
        return 1, len(lines)
    best_start, best_count = 1, -1
    flagged_sorted = sorted(set(flagged))
    for start in range(1, len(lines) - max_lines + 2, max(1, max_lines // 10)):
        end = start + max_lines - 1
        count = sum(1 for n in flagged_sorted if start <= n <= end)
        if count > best_count:
            best_start, best_count = start, count
    return best_start, best_start + max_lines - 1


def numbered(lines: Sequence[str], start: int, end: int) -> str:
    return "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(start, end + 1))


def flagged_lines(score: FileScore) -> list[int]:
    lines: list[int] = []
    for hit in score.signals:
        for ev in hit.evidence:
            match = re.search(r":(\d+)(?:-(\d+))?$", ev.location or "")
            if match:
                lines += range(int(match.group(1)), int(match.group(2) or match.group(1)) + 1)
    return lines


def resolve_pointers(
    pointers: Sequence[LinePointer], lines: Sequence[str], start: int, end: int
) -> list[QuotedEvidence]:
    """Turn line pointers into quoted evidence; drop lines outside the window or blank."""
    resolved = []
    for pointer in pointers:
        if start <= pointer.line <= end and lines[pointer.line - 1].strip():
            quote = _WS.sub(" ", lines[pointer.line - 1]).strip()
            resolved.append(
                QuotedEvidence(
                    line_start=pointer.line, line_end=pointer.line, quote=quote, why=pointer.why
                )
            )
    return resolved


def _cache_dir() -> Path:
    base = os.environ.get("MARGIN_CACHE_DIR") or str(Path.home() / ".cache" / "margin")
    return Path(base) / "judge"


def _cache_key(llm: LLMClient, window_text: str, variant: str) -> str:
    client = getattr(llm, "client", llm)  # sessions delegate to their provider client
    raw = "\n".join([client.provider, llm.model, PROMPT_VERSION, variant, window_text])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Judge:
    def __init__(
        self, llm: LLMClient, settings: JudgeSettings | None = None, *, cache: bool = True
    ):
        self.llm = llm
        self.settings = settings or JudgeSettings()
        self.cache = cache

    def _fit_window(
        self, lines: Sequence[str], flagged: Sequence[int], system: str
    ) -> tuple[int, int]:
        max_lines = self.settings.window_lines
        while True:
            start, end = choose_window(lines, flagged, max_lines)
            text = numbered(lines, start, end)
            # complete_structured appends the JSON schema and instructions to the system prompt
            overhead = len(SCHEMA_TEXT) + 400
            tokens = (len(system) + len(text) + overhead) / CHARS_PER_TOKEN
            if (
                tokens + self.settings.max_output_tokens <= self.settings.token_budget
                or max_lines <= 40
            ):
                return start, end
            max_lines = int(max_lines * 0.75)

    async def judge_file(
        self, path: str, text: str, flagged: Sequence[int], *, rubric_seed: int | None = None
    ) -> tuple[JudgeVerdict | None, bool, str | None]:
        """(verdict or None, cache hit, warning)."""
        lines = text.splitlines()
        if not lines:
            return None, False, None
        boundary = secrets.token_hex(6)
        open_marker, close_marker = f"<<<CODE-{boundary}", f"CODE-{boundary}>>>"
        system = build_system_prompt(open_marker, close_marker, rubric_seed)
        start, end = self._fit_window(lines, flagged, system)
        window_text = numbered(lines, start, end)
        variant = f"rubric:{rubric_seed}"
        key = _cache_key(self.llm, window_text, variant)
        cache_file = _cache_dir() / f"{key}.json"
        if self.cache and cache_file.exists():
            try:
                return JudgeVerdict.model_validate_json(cache_file.read_text("utf-8")), True, None
            except ValueError:
                cache_file.unlink(missing_ok=True)

        scope = "the whole file" if (start, end) == (1, len(lines)) else f"lines {start}-{end}"
        prompt = (
            f"File: {path} ({scope} of {len(lines)} lines)\n\n"
            f"{open_marker}\n{window_text}\n{close_marker}"
        )
        answer = await self.llm.complete_structured(
            JudgeAnswer, prompt, system=system, max_tokens=self.settings.max_output_tokens
        )
        verdict = JudgeVerdict(
            likelihood_ai=answer.likelihood_ai,
            confidence=answer.confidence,
            evidence_for=resolve_pointers(answer.evidence_for, lines, start, end),
            evidence_against=resolve_pointers(answer.evidence_against, lines, start, end),
            uncertainty_notes=answer.uncertainty_notes,
        )
        if not verdict.evidence_for and not verdict.evidence_against:
            return None, False, f"{path}: the judge pointed at no line of the file; ignored."
        if self.cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(verdict.model_dump_json(), encoding="utf-8")
        return verdict, False, None

    async def run(
        self,
        files: Sequence[tuple[FileScore, str]],
        *,
        max_seconds: float,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> JudgeRun:
        result = JudgeRun()
        t0 = time.monotonic()
        judged: list[tuple[FileScore, str]] = []
        # one warning per kind of error (numbers masked: memory figures vary between calls)
        failures: dict[str, tuple[str, list[str]]] = {}
        in_a_row = 0
        for index, (score, text) in enumerate(files):
            if cancelled is not None and cancelled():
                break
            if progress is not None:
                progress(index, len(files), score.path)
            if time.monotonic() - t0 > max_seconds:
                result.partial = True
                result.warnings.append(
                    f"The judge stopped after {max_seconds:.0f}s; "
                    f"{len(files) - len(judged)} sampled files were not judged."
                )
                break
            try:
                verdict, hit, warning = await self.judge_file(
                    score.path, text, flagged_lines(score)
                )
            except BudgetExceededError as exc:
                result.partial = True
                result.warnings.append(f"The judge stopped: {exc}")
                break
            except LLMError as exc:
                result.partial = True
                message = short_error(exc)
                key = re.sub(r"\d+(\.\d+)?", "#", message)
                failures.setdefault(key, (message, []))[1].append(score.path)
                in_a_row += 1
                if in_a_row >= MAX_FAILURES_IN_A_ROW:
                    result.warnings.append(
                        f"The judge stopped after {in_a_row} failed calls in a row; "
                        f"{len(files) - index - 1} sampled files were not judged."
                    )
                    break
                continue
            in_a_row = 0
            if warning:
                result.warnings.append(warning)
            if verdict is not None:
                result.verdicts[score.path] = verdict
                result.cache_hits += int(hit)
                judged.append((score, text))

        for message, paths in failures.values():
            where = paths[0] if len(paths) == 1 else f"{len(paths)} files ({paths[0]} and others)"
            result.warnings.insert(0, f"The judge failed on {where}: {message}")
        if cancelled is None or not cancelled():
            await self._agreement(result, judged, t0, max_seconds)
        return result

    async def _agreement(
        self, result: JudgeRun, judged: list[tuple[FileScore, str]], t0: float, max_seconds: float
    ) -> None:
        count = max(
            self.settings.agreement_min_files, round(len(judged) * self.settings.agreement_share)
        )
        subset = sorted(judged, key=lambda item: item[0].path)[: min(count, len(judged))]
        if len(subset) < self.settings.agreement_min_files:
            return
        diffs = []
        for score, text in subset:
            if time.monotonic() - t0 > max_seconds:
                break
            try:
                again, _, _ = await self.judge_file(
                    score.path, text, flagged_lines(score), rubric_seed=1
                )
            except LLMError:
                continue
            first = result.verdicts.get(score.path)
            if again is not None and first is not None:
                diffs.append(abs(again.likelihood_ai - first.likelihood_ai))
        if diffs:
            result.disagreement = round(sum(diffs) / len(diffs), 4)
            if result.disagreement > self.settings.disagreement_threshold:
                result.warnings.append(
                    f"The judge disagreed with itself by {result.disagreement:.2f} on average "
                    "when the rubric was reordered; its verdicts are unreliable here."
                )


def verdict_summary(verdicts: dict[str, JudgeVerdict]) -> str:
    if not verdicts:
        return "no verdicts"
    mean = sum(v.likelihood_ai for v in verdicts.values()) / len(verdicts)
    return json.dumps({"files": len(verdicts), "mean_likelihood_ai": round(mean, 3)})
