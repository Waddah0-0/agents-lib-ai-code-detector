"""Combine signal hits into file probabilities, a repository estimate, a range and a confidence.

Metadata markers never enter these numbers (see methodology notes in ``markers``).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ai_code_detector.methodology import Methodology
from ai_code_detector.models import Confidence, FileScore, SignalHit


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def family_scores(hits: Iterable[SignalHit], secondary_factor: float) -> dict[str, float]:
    """Per family: the strongest hit in full, the others reduced, clipped to [-1, 1]."""
    by_family: dict[str, list[float]] = defaultdict(list)
    for hit in hits:
        by_family[hit.family].append(hit.strength)
    scores = {}
    for family, strengths in by_family.items():
        ordered = sorted(strengths, key=abs, reverse=True)
        total = ordered[0] + secondary_factor * sum(ordered[1:])
        scores[family] = max(-1.0, min(1.0, total))
    return scores


def file_probability(hits: Sequence[SignalHit], methodology: Methodology) -> float:
    weights = methodology.file_weights
    scores = family_scores(hits, methodology.secondary_factor)
    z = methodology.bias + sum(weights.get(f, 0.0) * s for f, s in scores.items())
    return sigmoid(z)


def repo_shift(repo_hits: Sequence[SignalHit], methodology: Methodology) -> float:
    weights = methodology.repo_weights
    scores = family_scores(repo_hits, methodology.secondary_factor)
    return sum(weights.get(f, 0.0) * s for f, s in scores.items())


def _weighted_mean(files: Sequence[FileScore], default: float) -> float:
    if not files:
        return default
    weights = [math.log1p(f.lines) for f in files]
    return sum(w * f.probability for w, f in zip(weights, files, strict=True)) / sum(weights)


def repo_estimate(files: Sequence[FileScore], shift: float, methodology: Methodology) -> float:
    base = _weighted_mean(files, sigmoid(methodology.bias))
    return sigmoid(logit(base) + shift)


def estimate_range(
    files: Sequence[FileScore], shift: float, estimate: float, methodology: Methodology
) -> tuple[float, float]:
    """10th-90th percentile of the estimate over bootstrap resamples of the files."""
    cfg = methodology.section("bootstrap")
    if len(files) < int(cfg["min_files"]):
        return (round(min(0.1, estimate), 4), round(max(0.9, estimate), 4))
    rng = random.Random(int(cfg["seed"]))
    samples = sorted(
        repo_estimate([rng.choice(files) for _ in files], shift, methodology)
        for _ in range(int(cfg["resamples"]))
    )

    def quantile(q: float) -> float:
        return samples[min(len(samples) - 1, int(q * len(samples)))]

    low = min(quantile(float(cfg["low_quantile"])), estimate)
    high = max(quantile(float(cfg["high_quantile"])), estimate)
    return (round(low, 4), round(high, 4))


@dataclass(frozen=True)
class ConfidenceFacts:
    families: int
    files: int
    width: float
    supported_share: float


def confidence_level(
    facts: ConfidenceFacts, methodology: Methodology
) -> tuple[Confidence, list[str]]:
    c = methodology.section("confidence")
    high_checks = [
        (facts.families >= c["high_families"],
         f"evidence came from {facts.families} kinds of signals (high confidence needs "
         f"{c['high_families']})"),
        (facts.files >= c["high_files"],
         f"{facts.files} files could be analysed (high confidence needs {c['high_files']})"),
        (facts.width <= c["high_max_width"],
         f"the range is {facts.width:.2f} wide (high confidence needs "
         f"at most {c['high_max_width']})"),
        (facts.supported_share >= c["high_supported_share"],
         f"only {facts.supported_share:.0%} of the code is in a supported language"),
    ]  # fmt: skip
    medium_checks = [
        (facts.families >= c["medium_families"],
         f"evidence came from only {facts.families} kind(s) of signals"),
        (facts.files >= c["medium_files"], f"only {facts.files} files could be analysed"),
        (facts.width <= c["medium_max_width"], f"the range is wide ({facts.width:.2f})"),
    ]  # fmt: skip
    uncalibrated = (
        f"methodology {methodology.version} has not been calibrated against evaluation data yet"
    )
    if all(ok for ok, _ in high_checks):
        return ("high", []) if methodology.calibrated else ("medium", [uncalibrated])
    if all(ok for ok, _ in medium_checks):
        return "medium", [reason for ok, reason in high_checks if not ok]
    return "low", [reason for ok, reason in medium_checks if not ok]
