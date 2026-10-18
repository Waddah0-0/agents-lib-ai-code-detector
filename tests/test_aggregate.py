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

M = load_methodology()


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
