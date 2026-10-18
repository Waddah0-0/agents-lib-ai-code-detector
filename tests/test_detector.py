"""End to end: the detector on contrasting fixture repositories, through the registry."""

from __future__ import annotations

from pathlib import Path

from agents_core import AgentRegistry, get_agent, run_agent
from agents_core.reporting import to_markdown
from ai_code_detector import AICodeDetector


async def detect(target: Path, **extra: object) -> dict:  # type: ignore[type-arg]
    result = await run_agent("ai-code-detector", {"target": str(target), **extra})
    assert result.status in {"ok", "partial"}, result.error
    return {"result": result, "data": result.data}


def test_registered_via_entry_point() -> None:
    assert get_agent("ai-code-detector") is AICodeDetector
    assert AICodeDetector.info().primary_input == "target"


async def test_ai_style_repo_scores_higher_than_human_style(
    ai_repo: Path, human_repo: Path
) -> None:
    ai = (await detect(ai_repo))["data"]
    human = (await detect(human_repo))["data"]
    assert ai["estimate"] > 0.7
    assert human["estimate"] < 0.3
    assert ai["estimate_range"][0] <= ai["estimate"] <= ai["estimate_range"][1]
    assert ai["files_analyzed"] == 12 and human["files_analyzed"] == 12


async def test_markers_are_reported_but_not_scored(ai_repo: Path) -> None:
    data = (await detect(ai_repo))["data"]
    kinds = {(m["kind"], m["tool"]) for m in data["metadata_markers"]}
    assert kinds == {
        ("config_file", "Claude Code"),
        ("commit_trailer", "Claude"),
        ("generated_with_line", "Claude Code"),
    }
    assert all(f["ai_commit_touches"] == 1 for f in data["files"])
    assert not any(h["family"] == "metadata" for h in data["repo_signals"])


async def test_report_has_evidence_both_ways_and_the_caveat(human_repo: Path) -> None:
    result = (await detect(human_repo))["result"]
    titles = [f.title for f in result.findings]
    assert titles[0] == "Estimated likelihood of AI assistance"
    assert "Evidence toward human authorship" in titles
    human = next(f for f in result.findings if f.title == "Evidence toward human authorship")
    assert human.evidence and all((e.weight or 0) < 0 for e in human.evidence)
    markdown = to_markdown(result)
    assert "sole basis for accusing anyone" in markdown
    assert "AI tool usage confirmed by metadata" not in markdown


async def test_history_signals(ai_repo: Path, human_repo: Path) -> None:
    ai = (await detect(ai_repo))["data"]
    human = (await detect(human_repo))["data"]
    assert {h["signal_id"] for h in ai["repo_signals"]} == {"commit_shape", "readme_patterns"}
    depth = [h for h in human["repo_signals"] if h["signal_id"] == "history_depth"]
    assert depth and depth[0]["strength"] < 0


async def test_judge_request_is_partial_until_phase_4(ai_repo: Path) -> None:
    result = (await detect(ai_repo, use_llm_judge=True))["result"]
    assert result.status == "partial"
    assert any("LLM judge is not available" in w for w in result.warnings)


async def test_unparseable_files_make_the_result_partial(ai_repo: Path) -> None:
    (ai_repo / "broken.py").write_text("def (:\n" * 20, encoding="utf-8")
    result = (await detect(ai_repo))["result"]
    assert result.status == "partial"
    assert any("could not be read or parsed" in w for w in result.warnings)


async def test_tests_are_excluded_unless_asked(ai_repo: Path) -> None:
    tests = ai_repo / "tests"
    tests.mkdir()
    for i in range(3):
        (tests / f"test_m{i}.py").write_text((ai_repo / "src/app/module_0.py").read_text(), "utf-8")
    default = (await detect(ai_repo))["data"]
    assert default["files_analyzed"] == 12
    included = await detect(ai_repo, include_tests=True)
    assert included["data"]["files_analyzed"] == 15
    assert sum(f["is_test"] for f in included["data"]["files"]) == 3
    assert included["data"]["estimate"] == default["estimate"]  # tests don't move the estimate
    assert any(f.title == "Test files" for f in included["result"].findings)


async def test_no_git_history(tmp_path: Path, ai_code: str) -> None:
    for i in range(3):
        (tmp_path / f"m{i}.py").write_text(ai_code, encoding="utf-8")
    result = (await detect(tmp_path))["result"]
    # tmp_path may sit inside a git repo on some machines; either way the run must succeed
    assert result.data["files_analyzed"] == 3
    assert result.data["confidence"] == "low"


async def test_frozen_smoke_scores(ai_repo: Path, human_repo: Path) -> None:
    """CI regression guard: methodology v0 scores for the fixtures must not drift silently."""
    ai = (await detect(ai_repo))["data"]
    human = (await detect(human_repo))["data"]
    assert ai["methodology_version"] == "detector-0.1.0+methodology-v0+markers-v1"
    assert (round(ai["estimate"], 2), round(human["estimate"], 2)) == (SMOKE["ai"], SMOKE["human"])


SMOKE = {"ai": 0.83, "human": 0.27}  # methodology v0; update only with a new methodology version


def test_registry_lists_the_detector() -> None:
    assert "ai-code-detector" in AgentRegistry().ids()
