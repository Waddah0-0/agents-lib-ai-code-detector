"""The evaluation harness: manifest validation, metrics and an end-to-end run on local samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))

import dataset  # noqa: E402
import metrics  # noqa: E402
import run  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def test_auc() -> None:
    assert metrics.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert metrics.auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert metrics.auc([0.5, 0.5], [True, False]) == 0.5
    assert metrics.auc([0.5], [True]) is None


def test_rates_and_calibration() -> None:
    scores, labels = [0.9, 0.6, 0.4, 0.1], [True, False, True, False]
    assert metrics.rates_at(scores, labels, 0.5) == {"threshold": 0.5, "fpr": 0.5, "tpr": 0.5}
    assert metrics.expected_calibration_error([1.0, 0.0], [True, False]) == 0.0
    assert metrics.expected_calibration_error([0.9, 0.9], [False, False]) == 0.9


def test_manifest_validation(tmp_path: Path) -> None:
    good: dict[str, Any] = {
        "id": "x", "label": "human", "era": "pre2021", "provenance": "p",
        "provenance_confidence": "high", "split": "test",
        "source": {"kind": "git", "url": "https://github.com/octo/repo", "commit": "a" * 40},
    }  # fmt: skip
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(good) + "\n// comment\n", encoding="utf-8")
    assert dataset.load_manifest(manifest)[0].id == "x"

    for broken in (
        {**good, "source": {**good["source"], "commit": "abc"}},
        {**good, "source": {**good["source"], "url": "https://evil.com/x/y"}},
        {**good, "label": "robot"},
    ):
        manifest.write_text(json.dumps(broken) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="m.jsonl:1"):
            dataset.load_manifest(manifest)
    manifest.write_text((json.dumps(good) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        dataset.load_manifest(manifest)


def test_mixed_counts_as_ai_only_when_mostly_ai() -> None:
    base = {"id": "m", "era": "post2023", "provenance": "p", "provenance_confidence": "high",
            "split": "test", "source": {"kind": "local", "path": "x"}}  # fmt: skip
    assert dataset.Sample.model_validate({**base, "label": "mixed", "ai_fraction": 0.6}).is_ai
    assert not dataset.Sample.model_validate({**base, "label": "mixed", "ai_fraction": 0.3}).is_ai


async def test_end_to_end_on_local_samples(tmp_path: Path) -> None:
    for name, fixture, count in (("ai", "ai_style.py.txt", 6), ("human", "human_style.py.txt", 6)):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(count):
            (folder / f"m{i}.py").write_text((FIXTURES / fixture).read_text("utf-8"), "utf-8")
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        {"id": label, "label": label, "era": "post2023", "provenance": "fixture",
         "provenance_confidence": "high", "split": "train",
         "source": {"kind": "local", "path": label}}
        for label in ("ai", "human")
    ]  # fmt: skip
    manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    out = tmp_path / "results.json"

    report = await run.main_async(manifest, out, None, methodology="v0")

    assert json.loads(out.read_text("utf-8"))["metrics"] == report["metrics"]
    train = report["metrics"]["train"]
    assert train["repository_level"]["B2_signals"]["auc"] == 1.0
    assert train["repository_level"]["B0_constant_prior"]["auc"] == 0.5
    assert train["file_level"]["B2_signals"]["n"] == 12
    assert train["groups"]["language=python"]["n"] == 2
    skipped = await run.main_async(manifest, tmp_path / "r2.json", None, {"test"})
    assert skipped["samples"] == []  # test split only when asked
    assert report["metrics"]["errors"] == []
