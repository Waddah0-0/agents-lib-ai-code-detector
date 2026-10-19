"""Excerpts, the commit list and unscored files: what a report shows besides the scores."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ai_code_detector.markers import load_markers
from ai_code_detector.methodology import load_methodology
from ai_code_detector.models import (
    ExcerptMark,
    FileScore,
    JudgeVerdict,
    QuotedEvidence,
    SignalHit,
)
from ai_code_detector.report import (
    MAX_LINES_PER_FILE,
    build_excerpt,
    build_excerpts,
    commit_summaries,
    marks_for,
    parse_location,
    unscored,
)
from margin_core import run_agent
from margin_core.loaders import GitCommit
from margin_core.models import Evidence

M = load_methodology("v1")
TEXT = "\n".join(f"line {i}" for i in range(1, 201))


def mark(start: int, end: int | None = None, side: str = "for") -> ExcerptMark:
    return ExcerptMark(line_start=start, line_end=end or start, side=side, source="judge", note="n")  # type: ignore[arg-type]


def test_locations_must_point_into_the_file() -> None:
    assert parse_location("src/a.py:10-12", "src/a.py") == (10, 12)
    assert parse_location("src/a.py:7", "src/a.py") == (7, 7)
    assert parse_location("src/b.py:7", "src/a.py") is None
    assert parse_location("src/a.py", "src/a.py") is None
    assert parse_location(None, "src/a.py") is None


def test_nearby_marks_share_a_hunk_with_context() -> None:
    excerpt = build_excerpt("a.py", TEXT, [mark(10), mark(14), mark(100, 101)], budget=800)
    assert excerpt is not None
    assert [(h.start_line, len(h.lines)) for h in excerpt.hunks] == [(7, 11), (97, 8)]
    assert excerpt.hunks[0].lines[0] == "line 7"
    assert not excerpt.truncated and len(excerpt.marks) == 3


def test_excerpts_respect_the_line_budget() -> None:
    marks = [mark(i) for i in range(5, 200, 10)]  # 20 hunks of 7 lines
    excerpt = build_excerpt("a.py", TEXT, marks, budget=800)
    assert excerpt is not None and excerpt.truncated
    assert sum(len(h.lines) for h in excerpt.hunks) <= MAX_LINES_PER_FILE
    tight = build_excerpt("a.py", TEXT, marks, budget=10)
    assert tight is not None and sum(len(h.lines) for h in tight.hunks) <= 10
    assert all(m.line_start <= 14 for m in tight.marks)  # marks outside the shown lines dropped
    assert build_excerpt("a.py", "x = 1\n", [mark(50)], budget=800) is None


def hit(signal_id: str, family: str, location: str) -> SignalHit:
    evidence = [Evidence(kind="file", location=location, detail="x")]
    return SignalHit(signal_id=signal_id, signal_version="1", family=family, strength=0.5,
                     explanation="e", evidence=evidence)  # fmt: skip


def quote(line: int, end: int, why: str) -> QuotedEvidence:
    return QuotedEvidence(line_start=line, line_end=end, quote="q", why=why)


def test_marks_follow_the_methodology_weights() -> None:
    weight = M.file_weights
    family = next(f for f, w in weight.items() if w != 0)
    zero = next((f for f, w in weight.items() if w == 0), "unweighted_family")
    score = FileScore(
        path="a.py", language="python", lines=200, probability=0.6,
        signals=[hit("s1", family, "a.py:3"), hit("s2", zero, "a.py:9")],
        judge=JudgeVerdict(
            likelihood_ai=0.8, confidence=0.7,
            evidence_for=[quote(20, 20, "restates")], evidence_against=[quote(30, 31, "odd")],
        ),
    )  # fmt: skip
    marks = {m.line_start: m for m in marks_for(score, M)}
    expected = "for" if weight[family] > 0 else "against"
    assert marks[3].side == expected and marks[3].source == "s1"
    assert marks[9].side == "context"
    assert (marks[20].side, marks[30].side, marks[30].line_end) == ("for", "against", 31)

    excerpts = build_excerpts([score], M, lambda path: TEXT)
    assert len(excerpts) == 1 and excerpts[0].path == "a.py"


def commit(sha: str, message: str, parents: int = 1) -> GitCommit:
    when = datetime(2026, 1, 1, tzinfo=UTC)
    return GitCommit(
        sha=sha, author_name="Ann", author_email="ann@example.com", authored_at=when,
        committer_name="Ann", committer_email="ann@example.com", committed_at=when,
        parents=[f"p{i}" for i in range(parents)], message=message,
    )  # fmt: skip


def test_commit_summaries_flag_tools_and_large_commits_but_not_authors() -> None:
    commits = [
        commit("a1", "Add feature\n\nCo-Authored-By: Claude <noreply@anthropic.com>"),
        commit("b2", "Big import"),
        commit("c3", "Merge branch", parents=2),
    ]
    changed = {
        "a1": ["x.py"],
        "b2": [f"f{i}.py" for i in range(20)],
        "c3": [f"f{i}.py" for i in range(30)],
    }
    out = commit_summaries(commits, changed, load_markers(), large_commit_files=15)
    assert [c.sha for c in out] == ["a1", "b2", "c3"]
    assert out[0].tools == ["Claude"] and out[0].subject == "Add feature"
    assert out[1].large and not out[2].large and out[2].merge  # merges are never "large"
    assert "author" not in out[0].model_dump()


def test_unscored_files_are_capped() -> None:
    files, truncated = unscored([("a.go", "unsupported_language")])
    assert [(f.path, f.reason) for f in files] == [("a.go", "unsupported_language")]
    assert not truncated
    many, truncated = unscored((f"f{i}", "binary") for i in range(5000))
    assert len(many) == 3000 and truncated


async def test_detector_reports_excerpts_commits_and_unscored(ai_repo: Path) -> None:
    (ai_repo / "main.go").write_text("package main\n", encoding="utf-8")
    (ai_repo / "tiny.py").write_text("x = 1\n", encoding="utf-8")
    result = await run_agent("ai-code-detector", {"target": str(ai_repo), "use_llm_judge": False})
    assert result.status == "ok", result.warnings
    data = result.data
    assert data["commits_total"] == 1 and data["commits"][0]["tools"]
    reasons = {f["path"]: f["reason"] for f in data["unscored_files"]}
    assert reasons["main.go"] == "unsupported_language"
    assert reasons["tiny.py"] == "too_small"
    scored = {f["path"] for f in data["files"]}
    assert not scored & set(reasons)
    for excerpt in data["excerpts"]:
        assert excerpt["path"] in scored and excerpt["hunks"] and excerpt["marks"]
