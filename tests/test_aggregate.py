from __future__ import annotations

import pytest
from ai_code_detector.aggregate import (
    ConfidenceFacts,
    confidence_level,
    estimate_range,
    family_scores,
    file_probability,
    repo_estimate,
)
from ai_code_detector.methodology import Methodology, load_methodology
from ai_code_detector.models import FileScore, SignalHit

M = load_methodology("v0")  # mechanics; v1 has fitted numbers


def hit(family: str, strength: float) -> SignalHit:
    return SignalHit(
        signal_id="s", signal_version="1", family=family, strength=strength, explanation=""
    )


def file(p: float, lines: int = 100) -> FileScore:
    return FileScore(path=f"f{p}.py", language="python", lines=lines, probability=p)


def test_family_scores_count_correlated_signals_once() -> None:
    scores = family_scores([hit("comments", 0.8), hit("comments", 0.4), hit("naming", -0.3)], 0.25)
    assert scores == {"comments": pytest.approx(0.9), "naming": -0.3}
    assert family_scores([hit("x", 0.9), hit("x", 0.9)], 0.25)["x"] == 1.0  # clipped


def test_no_evidence_means_even_odds() -> None:
    assert file_probability([], M) == 0.5
    assert repo_estimate([], 0.0, M) == pytest.approx(0.5)


def test_directions() -> None:
    assert file_probability([hit("comments", 1.0)], M) > 0.7
    assert file_probability([hit("human_traces", -1.0)], M) < 0.3
    assert file_probability([hit("metadata", 1.0)], M) == 0.5  # markers have no weight


def test_repo_estimate_weights_by_size_and_shifts() -> None:
    files = [file(0.9, lines=1000), file(0.1, lines=10)]
    assert repo_estimate(files, 0.0, M) > 0.6
    assert repo_estimate(files, 1.0, M) > repo_estimate(files, 0.0, M)


def test_range_is_wide_with_few_files_and_contains_estimate() -> None:
    assert estimate_range([file(0.7)], 0.0, 0.7, M) == (0.1, 0.9)
    files = [file(p / 20) for p in range(1, 20)]
    est = repo_estimate(files, 0.0, M)
    low, high = estimate_range(files, 0.0, est, M)
    assert low <= est <= high and 0 < high - low < 0.5
    assert estimate_range(files, 0.0, est, M) == (low, high)  # deterministic


def test_confidence_levels() -> None:
    calibrated = Methodology(M.version, {**M.data, "calibrated": True})
    assert confidence_level(ConfidenceFacts(4, 30, 0.1, 0.9), calibrated) == ("high", [])
    level, reasons = confidence_level(ConfidenceFacts(4, 30, 0.1, 0.9), M)
    assert level == "medium" and "not been calibrated" in reasons[0]
    level, reasons = confidence_level(ConfidenceFacts(2, 10, 0.3, 0.9), M)
    assert level == "medium" and len(reasons) == 3
    level, reasons = confidence_level(ConfidenceFacts(1, 2, 0.8, 0.9), M)
    assert level == "low" and len(reasons) == 3


def test_repo_model_parameters_default_to_v0_behaviour() -> None:
    files = [file(0.8), file(0.6)]
    assert M.repo_intercept == 0.0 and M.repo_file_coefficient == 1.0
    fitted = Methodology(
        "x", {**M.data, "repo_model": {"intercept": -1.0, "file_coefficient": 2.0}}
    )
    assert repo_estimate(files, 0.0, fitted) != repo_estimate(files, 0.0, M)
    flat = Methodology("x", {**M.data, "repo_model": {"intercept": 0.0, "file_coefficient": 0.0}})
    assert repo_estimate(files, 0.0, flat) == pytest.approx(0.5)


def test_repository_level_judge_term() -> None:
    from ai_code_detector.aggregate import judge_shift
    from ai_code_detector.models import JudgeVerdict

    judged = file(0.5).model_copy(update={"judge": JudgeVerdict(likelihood_ai=0.9, confidence=1.0)})
    assert judge_shift([judged], M) == 0.0  # v0: no repository-level judge coefficient
    v1 = Methodology("x", {**M.data, "repo_model": {"judge_coefficient": 2.0}})
    assert judge_shift([judged, file(0.5)], v1) == pytest.approx(2.0 * 2.1972, rel=1e-3)
    assert judge_shift([file(0.5)], v1) == 0.0  # nothing judged


def test_range_reflects_how_few_files_the_judge_read() -> None:
    from ai_code_detector.aggregate import estimate_range, judge_shift, repo_estimate
    from ai_code_detector.models import JudgeVerdict

    v1 = Methodology("x", {**M.data, "repo_model": {"judge_coefficient": 2.0}})
    files = [file(0.55) for _ in range(20)]
    verdict = JudgeVerdict(likelihood_ai=0.9, confidence=1.0)
    files[0] = files[0].model_copy(update={"judge": verdict})
    estimate = repo_estimate(files, judge_shift(files, v1), v1)
    low, high = estimate_range(files, 0.0, estimate, v1)
    # identical signal scores used to give a zero-width range; one judged file must not
    assert low <= estimate <= high and high - low > 0.1
    # and when every file got the same verdict, the evidence agrees: the range is narrow
    same = [f.model_copy(update={"judge": verdict}) for f in files]
    everyone = repo_estimate(same, judge_shift(same, v1), v1)
    low, high = estimate_range(same, 0.0, everyone, v1)
    assert high - low < 0.02


def test_reported_range_is_at_least_as_wide_as_the_confidence_allows() -> None:
    from ai_code_detector.aggregate import widen_range

    assert widen_range(0.62, 0.63, 0.62, "low", M) == (0.47, 0.77)
    assert widen_range(0.62, 0.63, 0.62, "medium", M) == (0.55, 0.69)
    assert widen_range(0.62, 0.63, 0.62, "high", M) == (0.62, 0.63)
    assert widen_range(0.2, 0.9, 0.5, "low", M) == (0.2, 0.9)  # already wide: unchanged
    assert widen_range(0.95, 0.97, 0.96, "low", M) == (0.81, 1.0)  # clipped to [0, 1]
