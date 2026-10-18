"""Fit the detector's weights on the TRAIN split and compare candidates on VALIDATION.

Usage::

    uv run python agents/ai_code_detector/eval/fit.py \\
        --results agents/ai_code_detector/eval/results/v1_dev_methodology_v0.json

Two logistic regressions (L2-regularised, fitted with Newton's method; no dependencies):

1. File model: P(AI | file) from the file-level family scores. Every training repository
   gets the same total weight (its files share it), and both classes get the same total
   weight, so big repositories and the larger class cannot dominate.
2. Repository model: P(AI | repo) from the file model's size-weighted mean (as log-odds)
   plus the repository-level family scores.

The test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import summarize  # noqa: E402

FILE_FAMILIES = ("comments", "naming", "human_traces")
REPO_FAMILIES = ("history", "docs")
FILE_CAP = 50


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (tiny systems only)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_logistic(
    rows: Sequence[Sequence[float]],
    labels: Sequence[bool],
    weights: Sequence[float],
    l2: float = 1.0,
    iterations: int = 50,
) -> list[float]:
    """Coefficients [intercept, w1, ...]; the intercept is not regularised."""
    k = len(rows[0]) + 1
    beta = [0.0] * k
    xs = [[1.0, *row] for row in rows]
    for _ in range(iterations):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for x, y, w in zip(xs, labels, weights, strict=True):
            p = sigmoid(sum(b * v for b, v in zip(beta, x, strict=True)))
            for i in range(k):
                grad[i] += w * (p - float(y)) * x[i]
                for j in range(k):
                    hess[i][j] += w * p * (1 - p) * x[i] * x[j]
        for i in range(1, k):
            grad[i] += l2 * beta[i]
            hess[i][i] += l2
        step = _solve(hess, grad)
        beta = [b - s for b, s in zip(beta, step, strict=True)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return [round(b, 4) for b in beta]


def _capped(record: dict[str, Any]) -> list[dict[str, Any]]:
    import hashlib

    files = sorted(record["files"], key=lambda f: hashlib.sha1(f["path"].encode()).hexdigest())
    return files[:FILE_CAP]


def _file_features(f: dict[str, Any]) -> list[float]:
    return [float(f["families"].get(name, 0.0)) for name in FILE_FAMILIES]


def fit_file_model(train: list[dict[str, Any]], l2: float) -> list[float]:
    rows, labels, weights = [], [], []
    class_totals = {True: sum(r["is_ai"] for r in train), False: sum(not r["is_ai"] for r in train)}
    for r in train:
        files = _capped(r)
        for f in files:
            rows.append(_file_features(f))
            labels.append(r["is_ai"])
            weights.append(1.0 / len(files) / class_totals[r["is_ai"]] * len(train) / 2)
    return fit_logistic(rows, labels, weights, l2)


def file_probability(beta: list[float], f: dict[str, Any]) -> float:
    return sigmoid(beta[0] + sum(b * x for b, x in zip(beta[1:], _file_features(f), strict=True)))


def mean_file_logit(beta: list[float], r: dict[str, Any]) -> float:
    files = r["files"]
    if not files:
        return 0.0
    weights = [math.log1p(f["lines"]) for f in files]
    probs = [file_probability(beta, f) for f in files]
    return logit(sum(w * p for w, p in zip(weights, probs, strict=True)) / sum(weights))


def _repo_features(file_beta: list[float], r: dict[str, Any]) -> list[float]:
    fam = r.get("repo_families", {})
    return [mean_file_logit(file_beta, r), *(float(fam.get(n, 0.0)) for n in REPO_FAMILIES)]


def fit_repo_model(train: list[dict[str, Any]], file_beta: list[float], l2: float) -> list[float]:
    rows = [_repo_features(file_beta, r) for r in train]
    labels = [r["is_ai"] for r in train]
    n_ai = sum(labels)
    weights = [len(train) / 2 / (n_ai if y else len(train) - n_ai) for y in labels]
    return fit_logistic(rows, labels, weights, l2)


def repo_probability(file_beta: list[float], repo_beta: list[float], r: dict[str, Any]) -> float:
    x = _repo_features(file_beta, r)
    return sigmoid(repo_beta[0] + sum(b * v for b, v in zip(repo_beta[1:], x, strict=True)))


def evaluate(
    records: list[dict[str, Any]], file_beta: list[float], repo_beta: list[float]
) -> dict[str, Any]:
    labels = [r["is_ai"] for r in records]
    repo_scores = [repo_probability(file_beta, repo_beta, r) for r in records]
    file_scores = [file_probability(file_beta, f) for r in records for f in _capped(r)]
    file_labels = [r["is_ai"] for r in records for _ in _capped(r)]
    return {
        "repository": summarize(repo_scores, labels, (0.5, 0.6, 0.7)),
        "file": summarize(file_scores, file_labels, (0.5, 0.6, 0.7)),
        "per_repo": {r["id"]: round(s, 3) for r, s in zip(records, repo_scores, strict=True)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit detector weights on the train split.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="write the fitted weights as JSON")
    args = parser.parse_args(argv)
    report = json.loads(args.results.read_text("utf-8"))
    records = [r for r in report["samples"] if r["status"] != "error" and r["label"] != "mixed"]
    if any(r["split"] == "test" for r in records):
        print("note: test-split records present in the results; they are ignored here")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]

    file_beta = fit_file_model(train, args.l2)
    repo_beta = fit_repo_model(train, file_beta, args.l2)
    fitted = {
        "file_model": dict(zip(("intercept", *FILE_FAMILIES), file_beta, strict=True)),
        "repo_model": dict(
            zip(("intercept", "mean_file_logit", *REPO_FAMILIES), repo_beta, strict=True)
        ),
        "l2": args.l2,
        "train": evaluate(train, file_beta, repo_beta),
        "val": evaluate(val, file_beta, repo_beta),
    }
    print(json.dumps({k: v for k, v in fitted.items() if k not in {"train", "val"}}, indent=2))
    for split in ("train", "val"):
        rep, fil = fitted[split]["repository"], fitted[split]["file"]
        print(f"{split}: repo AUC {rep['auc']} ECE {rep['ece']} at0.5 {rep['at_threshold'][0]} "
              f"| file AUC {fil['auc']}")  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
