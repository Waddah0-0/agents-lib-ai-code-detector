"""Evaluation metrics, implemented directly so the harness has no extra dependencies."""

from __future__ import annotations

from collections.abc import Sequence


def auc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """Area under the ROC curve (probability a positive outscores a negative; ties count half).

    ``None`` when either class is missing.
    """
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def rates_at(
    scores: Sequence[float], positives: Sequence[bool], threshold: float
) -> dict[str, float | None]:
    """False-positive rate on negatives and true-positive rate on positives at ``threshold``."""
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    return {
        "threshold": threshold,
        "fpr": round(sum(s >= threshold for s in neg) / len(neg), 4) if neg else None,
        "tpr": round(sum(s >= threshold for s in pos) / len(pos), 4) if pos else None,
    }


def expected_calibration_error(
    scores: Sequence[float], positives: Sequence[bool], bins: int = 10
) -> float | None:
    """Mean gap between predicted probability and observed frequency, weighted by bin size."""
    if not scores:
        return None
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (s, p)
            for s, p in zip(scores, positives, strict=True)
            if low <= s < high or (b == bins - 1 and s == 1.0)
        ]
        if members:
            mean_score = sum(s for s, _ in members) / len(members)
            frequency = sum(1 for _, p in members if p) / len(members)
            total += len(members) * abs(mean_score - frequency)
    return round(total / len(scores), 4)


def summarize(
    scores: Sequence[float], positives: Sequence[bool], thresholds: Sequence[float]
) -> dict[str, object]:
    return {
        "n": len(scores),
        "positives": sum(positives),
        "auc": auc(scores, positives),
        "ece": expected_calibration_error(scores, positives),
        "at_threshold": [rates_at(scores, positives, t) for t in thresholds],
    }
