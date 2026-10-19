"""What a report needs besides the scores: code excerpts around the evidence, a commit list
for the timeline, and the files that have no score. Everything here is capped, so a large
repository still produces a result of a sensible size."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from agents_core.loaders import GitCommit

from ai_code_detector.markers import MarkerSet, commit_marker_tools
from ai_code_detector.methodology import Methodology
from ai_code_detector.models import (
    CommitSummary,
    Excerpt,
    ExcerptHunk,
    ExcerptMark,
    FileScore,
    QuotedEvidence,
    Side,
    UnscoredFile,
)

CONTEXT_LINES = 3  # lines shown around each marked span
MERGE_GAP = 2  # hunks closer than this are joined
MAX_EXCERPT_FILES = 30
MAX_LINES_PER_FILE = 80
MAX_LINES_TOTAL = 800
MAX_LINE_CHARS = 240
MAX_NOTE_CHARS = 120
MAX_COMMITS_REPORTED = 500
MAX_SUBJECT_CHARS = 120
MAX_UNSCORED = 3000

_LOCATION = re.compile(r"^(?P<path>.+):(?P<start>\d+)(?:-(?P<end>\d+))?$")


def parse_location(location: str | None, path: str) -> tuple[int, int] | None:
    """``"src/a.py:10-12"`` -> ``(10, 12)`` when it points into ``path``."""
    if not location:
        return None
    match = _LOCATION.match(location)
    if match is None or match["path"] != path:
        return None
    start = int(match["start"])
    end = int(match["end"]) if match["end"] else start
    return (start, max(start, end))


def _side(weight: float, strength: float) -> Side:
    if weight == 0 or strength == 0:
        return "context"
    return "for" if weight * strength > 0 else "against"


def marks_for(score: FileScore, methodology: Methodology) -> list[ExcerptMark]:
    """Every located piece of evidence in one file: the judge's quotes and the signals'."""
    marks: list[ExcerptMark] = []
    if score.judge is not None:
        sides: list[tuple[Side, list[QuotedEvidence]]] = [
            ("for", score.judge.evidence_for), ("against", score.judge.evidence_against),
        ]  # fmt: skip
        for side, quotes in sides:
            for q in quotes:
                marks.append(ExcerptMark(
                    line_start=q.line_start, line_end=max(q.line_start, q.line_end),
                    side=side, source="judge", note=q.why[:MAX_NOTE_CHARS],
                ))  # fmt: skip
    weights = methodology.file_weights
    for hit in score.signals:
        side = _side(weights.get(hit.family, 0.0), hit.strength)
        for ev in hit.evidence:
            span = parse_location(ev.location, score.path)
            if span is not None:
                marks.append(ExcerptMark(
                    line_start=span[0], line_end=span[1], side=side,
                    source=hit.signal_id, note=ev.detail[:MAX_NOTE_CHARS],
                ))  # fmt: skip
    return sorted(marks, key=lambda m: (m.line_start, m.line_end))


def _hunk_ranges(marks: list[ExcerptMark], n_lines: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for m in marks:
        start = max(1, m.line_start - CONTEXT_LINES)
        end = min(n_lines, m.line_end + CONTEXT_LINES)
        if start > end:
            continue
        if ranges and start <= ranges[-1][1] + MERGE_GAP:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def build_excerpt(path: str, text: str, marks: list[ExcerptMark], budget: int) -> Excerpt | None:
    """The hunks of ``text`` around ``marks``, within ``budget`` lines (and the per-file cap)."""
    lines = text.splitlines()
    marks = [m for m in marks if m.line_start <= len(lines)]
    if not marks:
        return None
    limit = min(budget, MAX_LINES_PER_FILE)
    hunks: list[ExcerptHunk] = []
    used = 0
    truncated = False
    for start, end in _hunk_ranges(marks, len(lines)):
        size = end - start + 1
        if used + size > limit:
            truncated = True
            if used == 0:  # always show the start of the first hunk
                end = start + limit - 1
                size = limit
            else:
                break
        hunks.append(ExcerptHunk(
            start_line=start,
            lines=[line[:MAX_LINE_CHARS] for line in lines[start - 1 : end]],
        ))  # fmt: skip
        used += size
        if truncated:
            break
    shown = [(h.start_line, h.start_line + len(h.lines) - 1) for h in hunks]
    kept = [m for m in marks if any(a <= m.line_start <= b for a, b in shown)]
    truncated = truncated or len(kept) < len(marks)
    return Excerpt(path=path, hunks=hunks, marks=kept, truncated=truncated)


def build_excerpts(
    files: list[FileScore], methodology: Methodology, read: Callable[[str], str]
) -> list[Excerpt]:
    """Excerpts for the files the judge read, then the files with the most located evidence."""
    candidates = [(f, marks_for(f, methodology)) for f in files]
    candidates = [(f, m) for f, m in candidates if m]
    candidates.sort(key=lambda c: (c[0].judge is None, -c[0].probability))
    excerpts: list[Excerpt] = []
    budget = MAX_LINES_TOTAL
    for score, marks in candidates[:MAX_EXCERPT_FILES]:
        if budget <= 0:
            break
        try:
            text = read(score.path)
        except OSError:
            continue
        excerpt = build_excerpt(score.path, text, marks, budget)
        if excerpt is not None:
            excerpts.append(excerpt)
            budget -= sum(len(h.lines) for h in excerpt.hunks)
    return excerpts


def commit_summaries(
    commits: list[GitCommit],
    changed: dict[str, list[str]],
    markers: MarkerSet,
    large_commit_files: int,
) -> list[CommitSummary]:
    """The newest commits (git order), with marker tools and a large-commit flag."""
    result = []
    for c in commits[:MAX_COMMITS_REPORTED]:
        n_files = len(changed.get(c.sha, []))
        merge = len(c.parents) > 1
        result.append(CommitSummary(
            sha=c.sha, date=c.authored_at, subject=c.subject[:MAX_SUBJECT_CHARS],
            files_changed=n_files, merge=merge,
            tools=sorted({tool for _, tool in commit_marker_tools(c, markers)}),
            large=not merge and n_files >= large_commit_files,
        ))  # fmt: skip
    return result


def unscored(entries: Iterable[tuple[str, str]]) -> tuple[list[UnscoredFile], bool]:
    """``(path, reason)`` pairs as models, capped; the flag says whether some were dropped."""
    files: list[UnscoredFile] = []
    for path, reason in entries:
        if len(files) == MAX_UNSCORED:
            return files, True
        files.append(UnscoredFile(path=path, reason=reason))
    return files, False
