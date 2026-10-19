"""The detector pipeline: ingest -> markers -> repo signals -> file signals -> aggregate -> report.

Steps match PLAN §9.3. The LLM judge (steps 5-6) arrives in phase 4.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from agents_core import Evidence, Finding, RunContext
from agents_core.loaders import FileInfo, Workspace

from ai_code_detector.aggregate import (
    ConfidenceFacts,
    confidence_level,
    estimate_range,
    file_probability,
    judge_shift,
    repo_estimate,
    repo_shift,
    widen_range,
)
from ai_code_detector.judge.llm_judge import Judge, JudgeRun, JudgeSettings
from ai_code_detector.judge.sampler import sample_files
from ai_code_detector.markers import MarkerSet, detect_commit_markers, detect_config_files
from ai_code_detector.methodology import Methodology
from ai_code_detector.models import DetectorInput, DetectorOutput, FileScore, SignalHit
from ai_code_detector.report import build_excerpts, commit_summaries, unscored
from ai_code_detector.signals import FILE_SIGNALS, REPO_SIGNALS, RepoContext
from ai_code_detector.source import is_supported, parse_source

MAX_COMMITS = 5000
FILE_BATCH = 50
SUPPORTED_LANGUAGES = {"python", "javascript", "typescript"}
CODE_LANGUAGES = {
    "python", "javascript", "typescript", "java", "kotlin", "scala", "go", "rust", "ruby", "php",
    "swift", "c", "cpp", "csharp", "fsharp", "dart", "lua", "r", "objective-c", "elixir",
    "erlang", "haskell", "clojure", "julia", "zig", "shell", "powershell", "sql", "vue", "svelte",
}  # fmt: skip
CAVEAT = (
    "This is an estimate, not proof: clean human code can look machine-made and edited AI code "
    "can look human. Do not use it as the sole basis for accusing anyone."
)


@dataclass
class Detection:
    output: DetectorOutput
    summary: str
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)
    partial: bool = False


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name
    return (
        any(
            part in {"test", "tests", "testing", "__tests__", "spec", "e2e"}
            for part in p.parts[:-1]
        )
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or any(f".{kind}." in name for kind in ("test", "spec"))  # foo.test.ts, bar.spec.js
    )


def _analyse_file(ws: Workspace, info: FileInfo, methodology: Methodology) -> FileScore | str:
    """A scored file, or the reason it was skipped."""
    try:
        text = ws.read_text(info)
    except OSError:
        return "unreadable"
    source = parse_source(info.path, text)
    if source is None:
        return "unparseable"
    if source.code_lines < methodology.min_code_lines:
        return "too_small"
    hits = [hit for signal in FILE_SIGNALS for hit in signal.evaluate(source, methodology)]
    return FileScore(
        path=info.path,
        language=info.language,
        lines=source.code_lines,
        is_test=is_test_path(info.path),
        probability=round(file_probability(hits, methodology), 4),
        signals=hits,
    )


async def _run_judge(
    ws: Workspace,
    params: DetectorInput,
    ctx: RunContext,
    methodology: Methodology,
    scored: list[FileScore],
) -> JudgeRun:
    """Judge a sample of files and fold the verdicts into their probabilities."""
    if ctx.llm is None:
        return JudgeRun(
            warnings=["The LLM judge was requested, but no model is configured."], partial=True
        )
    cfg = methodology.section("judge")
    settings = JudgeSettings(
        window_lines=int(cfg["window_lines"]),
        token_budget=int(cfg["token_budget"]),
        max_output_tokens=int(cfg["max_output_tokens"]),
        agreement_share=float(cfg["agreement_share"]),
        agreement_min_files=int(cfg["agreement_min_files"]),
        disagreement_threshold=float(cfg["disagreement_threshold"]),
    )
    sample = sample_files(
        scored, max_files=params.max_files_for_judge, min_lines=int(cfg["min_lines"]),
        seed_text=ws.source,
    )  # fmt: skip
    if not sample:
        return JudgeRun(warnings=["No file was large enough for the LLM judge."])
    texts = await asyncio.to_thread(lambda: [ws.read_text(f.path) for f in sample])
    llm = ctx.llm.session(max_cost_usd=params.max_cost_usd)

    def progress(index: int, total: int, path: str) -> None:
        # the judge is the slow part (minutes), so it gets most of the bar
        ctx.report_progress(
            "judge", f"Judging {path} ({index + 1}/{total})", 0.35 + 0.55 * index / max(total, 1)
        )

    run = await Judge(llm, settings).run(
        list(zip(sample, texts, strict=True)),
        max_seconds=params.max_judge_seconds,
        cancelled=lambda: ctx.cancel_token.cancelled,
        progress=progress,
    )
    for f in scored:
        verdict = run.verdicts.get(f.path)
        if verdict is not None:
            f.judge = verdict
            f.probability = round(
                file_probability(f.signals, methodology, verdict.likelihood_ai, verdict.confidence),
                4,
            )
    if not run.verdicts:
        run.warnings.append("The LLM judge produced no usable verdicts.")
        run.partial = True
    return run


def _ranked_evidence(
    repo_hits: list[SignalHit],
    files: list[FileScore],
    methodology: Methodology,
    toward_ai: bool,
    limit: int = 5,
) -> list[Evidence]:
    weights = {**methodology.file_weights, **methodology.repo_weights}
    candidates: list[tuple[float, SignalHit, str | None]] = []
    for hit in repo_hits:
        candidates.append((hit.strength * weights.get(hit.family, 0.0), hit, None))
    for f in files:
        for hit in f.signals:
            candidates.append((hit.strength * weights.get(hit.family, 0.0), hit, f.path))
    chosen = [c for c in candidates if (c[0] > 0) == toward_ai and c[0] != 0]
    chosen.sort(key=lambda c: abs(c[0]), reverse=True)
    evidence = []
    for score, hit, path in chosen[:limit]:
        first = hit.evidence[0] if hit.evidence else None
        evidence.append(
            Evidence(
                kind=first.kind if first else "file",
                location=(first.location if first else None) or path,
                detail=hit.explanation,
                weight=round(score, 3),
            )
        )
    return evidence


async def detect(
    ws: Workspace,
    params: DetectorInput,
    ctx: RunContext,
    methodology: Methodology,
    markers: MarkerSet,
    methodology_version: str,
) -> Detection:
    warnings: list[str] = []
    partial = False

    # 2. metadata markers and history
    ctx.report_progress("markers", "Reading history and looking for AI tool markers", 0.1)
    commits = []
    changed: dict[str, list[str]] = {}
    if ws.git is not None and ws.git.head is not None:
        commits = await ws.git.commits(max_count=MAX_COMMITS)
        changed = await ws.git.changed_files(max_count=MAX_COMMITS)
    else:
        warnings.append("No git history was available, so history signals were skipped.")
    truncated = len(commits) >= MAX_COMMITS
    if truncated:
        warnings.append(f"Only the newest {MAX_COMMITS} commits were read.")
    found_markers = detect_config_files(ws, markers)
    commit_markers, marked_shas = detect_commit_markers(commits, markers)
    found_markers += commit_markers
    ctx.check_cancelled()

    # 3. repository signals
    ctx.report_progress("repo_signals", "Evaluating the repository", 0.2)
    repo_ctx = RepoContext(ws, methodology, commits, changed, history_truncated=truncated)
    repo_hits = [hit for signal in REPO_SIGNALS for hit in signal.evaluate(repo_ctx)]
    ctx.check_cancelled()

    # 4. file signals (in batches off the event loop, so progress and cancellation keep working)
    candidates = [
        f for f in ws.files
        if f.language in SUPPORTED_LANGUAGES and is_supported(f.path)
        and (params.include_tests or not is_test_path(f.path))
    ]  # fmt: skip
    scored: list[FileScore] = []
    skip_reasons: Counter[str] = Counter()
    skipped_paths: dict[str, str] = {}
    for start in range(0, len(candidates), FILE_BATCH):
        batch = candidates[start : start + FILE_BATCH]
        ctx.report_progress(
            "file_signals",
            f"Analysing files {start + 1}-{start + len(batch)} of {len(candidates)}",
            0.2 + 0.15 * start / max(len(candidates), 1),
        )
        results = await asyncio.to_thread(
            lambda b=batch: [_analyse_file(ws, info, methodology) for info in b]  # type: ignore[misc]
        )
        for info, outcome in zip(batch, results, strict=True):
            if isinstance(outcome, str):
                skip_reasons[outcome] += 1
                skipped_paths[info.path] = outcome
            else:
                scored.append(outcome)
        ctx.check_cancelled()

    touches: Counter[str] = Counter()
    for sha in marked_shas:
        touches.update(changed.get(sha, []))
    for f in scored:
        f.ai_commit_touches = touches.get(f.path, 0)

    if skip_reasons["unparseable"] or skip_reasons["unreadable"]:
        n = skip_reasons["unparseable"] + skip_reasons["unreadable"]
        warnings.append(f"{n} source files could not be read or parsed and were skipped.")
        partial = True
    # 5-6. LLM judge on a sample of files
    # Explicit True: required (partial if it cannot run). None (default): run it when a model
    # is configured, otherwise fall back to signals only. False: never.
    judge_disagreement = False
    explicit = params.use_llm_judge is True
    if explicit or (params.use_llm_judge is None and ctx.llm is not None):
        judge_outcome = await _run_judge(ws, params, ctx, methodology, scored)
        warnings += judge_outcome.warnings
        partial = partial or (explicit and judge_outcome.partial)
        judge_disagreement = judge_outcome.disagreement is not None and (
            judge_outcome.disagreement
            > float(methodology.section("judge")["disagreement_threshold"])
        )
    ctx.check_cancelled()  # a cancelled judge stops early; don't build a report from it
    judged = any(f.judge is not None for f in scored if not f.is_test)
    if not judged and "repo_model_signals_only" in methodology.data:
        methodology = Methodology(
            methodology.version,
            {**methodology.data, "repo_model": methodology.data["repo_model_signals_only"]},
        )

    # 7. aggregate
    ctx.report_progress("aggregate", "Combining the evidence", 0.95)
    main_files = [f for f in scored if not f.is_test]
    repo_level = repo_shift(repo_hits, methodology)
    shift = repo_level + judge_shift(main_files, methodology)
    estimate = round(repo_estimate(main_files, shift, methodology), 4)
    low, high = estimate_range(main_files, repo_level, estimate, methodology)

    languages = Counter(f.language for f in ws.files if f.language in CODE_LANGUAGES)
    supported = {k: v for k, v in languages.items() if k in SUPPORTED_LANGUAGES}
    unsupported = {k: v for k, v in languages.items() if k not in SUPPORTED_LANGUAGES}
    code_bytes = sum(f.size for f in ws.files if f.language in CODE_LANGUAGES)
    supported_bytes = sum(f.size for f in ws.files if f.language in SUPPORTED_LANGUAGES)
    supported_share = supported_bytes / code_bytes if code_bytes else 0.0
    families = {h.family for h in repo_hits} | {h.family for f in main_files for h in f.signals}
    confidence, reasons = confidence_level(
        ConfidenceFacts(
            len(families), len(main_files), high - low, supported_share, judge_disagreement
        ),
        methodology,
    )
    cap = methodology.section("confidence").get("signals_only_max")
    if not judged and cap == "low" and confidence != "low":
        confidence = "low"
        reasons = [
            "the LLM judge did not run, and the heuristic signals alone separate human from AI "
            "code poorly"
        ]
    low, high = widen_range(low, high, estimate, confidence, methodology)

    # 8. what the report shows besides the scores
    scored_paths = {f.path for f in scored}
    excerpts = await asyncio.to_thread(build_excerpts, scored, methodology, ws.read_text)

    def unscored_reason(info: FileInfo) -> str:
        if info.path in skipped_paths:
            return skipped_paths[info.path]
        if info.language in SUPPORTED_LANGUAGES and is_supported(info.path):
            return "test_file"
        return "unsupported_language"

    unscored_files, unscored_truncated = unscored(
        [(f.path, unscored_reason(f)) for f in ws.files if f.path not in scored_paths]
        + [(s.path, s.reason) for s in ws.skipped]
    )
    large_files = int(methodology.thresholds_for("commit_shape")["large_commit_files"])

    output = DetectorOutput(
        methodology_version=methodology_version,
        estimate=estimate,
        estimate_range=(low, high),
        confidence=confidence,
        confidence_reasons=reasons,
        metadata_markers=found_markers,
        repo_signals=repo_hits,
        files=sorted(scored, key=lambda f: f.probability, reverse=True),
        files_analyzed=len(scored),
        files_skipped=len(ws.skipped) + sum(skip_reasons.values()),
        languages=dict(supported),
        unsupported_languages=dict(unsupported),
        excerpts=excerpts,
        commits=commit_summaries(commits, changed, markers, large_files),
        commits_total=len(commits),
        unscored_files=unscored_files,
        unscored_truncated=unscored_truncated,
    )
    findings = _findings(output, repo_hits, main_files, scored, methodology, supported_share)
    summary = (
        f"Estimated likelihood that this code was written with AI assistance: {estimate:.2f} "
        f"(range {low:.2f}-{high:.2f}, {confidence} confidence), from {len(main_files)} source "
        f"files and {len(commits)} commits."
    )
    if found_markers:
        tools = sorted({m.tool for m in found_markers})
        summary += f" Metadata confirms AI tool usage ({', '.join(tools)})."
    summary += " " + CAVEAT
    return Detection(output, summary, findings, warnings, partial)


def _findings(
    output: DetectorOutput,
    repo_hits: list[SignalHit],
    main_files: list[FileScore],
    scored: list[FileScore],
    methodology: Methodology,
    supported_share: float,
) -> list[Finding]:
    low, high = output.estimate_range
    estimate_summary = (
        f"{output.estimate:.2f}, plausibly between {low:.2f} and {high:.2f}; "
        f"{output.confidence} confidence."
    )
    if output.confidence_reasons:
        estimate_summary += (
            " Confidence is limited because " + "; ".join(output.confidence_reasons) + "."
        )
    findings = [
        Finding(
            title="Estimated likelihood of AI assistance",
            severity_or_score=output.estimate,
            summary=estimate_summary,
            evidence=[
                Evidence(kind="metric", detail=f"range {low:.2f}-{high:.2f}"),
                Evidence(kind="metric", detail=f"methodology {output.methodology_version}"),
            ],
        )
    ]
    if output.metadata_markers:
        findings.append(
            Finding(
                title="AI tool usage confirmed by metadata",
                summary=(
                    "These traces prove that AI tools were used in this repository. They do not "
                    "show how much of the code the tools wrote, so they are not part of the "
                    "estimate."
                ),
                evidence=[
                    Evidence(
                        kind="config" if m.kind == "config_file" else "commit",
                        location=m.location,
                        detail=m.detail,
                    )
                    for m in output.metadata_markers
                ],
            )
        )
    context = [h for h in repo_hits if not methodology.repo_weights.get(h.family, 0.0)]
    if context:
        findings.append(
            Finding(
                title="Repository context (not part of the estimate)",
                summary=(
                    "Shown for context only. In evaluation, commit-history and README style "
                    "mostly separated old projects from new ones, not human from AI authors."
                ),
                evidence=[
                    Evidence(
                        kind=h.evidence[0].kind if h.evidence else "metric",
                        location=h.evidence[0].location if h.evidence else None,
                        detail=h.explanation,
                    )
                    for h in context
                ],
            )
        )
    toward_ai = _ranked_evidence(repo_hits, main_files, methodology, toward_ai=True)
    toward_human = _ranked_evidence(repo_hits, main_files, methodology, toward_ai=False)
    findings.append(
        Finding(
            title="Evidence toward AI assistance",
            summary="The strongest signs of AI-written code." if toward_ai else "None found.",
            evidence=toward_ai,
        )
    )
    findings.append(
        Finding(
            title="Evidence toward human authorship",
            summary="The strongest signs of people at work." if toward_human else "None found.",
            evidence=toward_human,
        )
    )
    tests = [f for f in scored if f.is_test]
    if tests:
        mean = sum(f.probability for f in tests) / len(tests)
        findings.append(
            Finding(
                title="Test files",
                severity_or_score=round(mean, 4),
                summary=f"{len(tests)} test files average {mean:.2f}; they are not part of the "
                "estimate.",
            )
        )
    if output.unsupported_languages and supported_share < 0.8:
        langs = ", ".join(f"{k} ({v})" for k, v in sorted(output.unsupported_languages.items()))
        findings.append(
            Finding(
                title="Coverage",
                summary=(
                    f"Only {supported_share:.0%} of the code is in a language analysed file by "
                    f"file (Python, JavaScript, TypeScript). Not analysed: {langs}."
                ),
            )
        )
    findings.append(Finding(title="How to read this result", summary=CAVEAT))
    return findings
