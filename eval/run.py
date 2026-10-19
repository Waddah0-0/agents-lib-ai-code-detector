"""Run the detector over an evaluation dataset and compute baseline metrics.

Usage (from the repository root)::

    uv run python agents/ai_code_detector/eval/run.py \\
        --manifest agents/ai_code_detector/eval/datasets/v0/manifest.jsonl \\
        --out agents/ai_code_detector/eval/results/v0.json

Baselines (PLAN §9.7): B0 constant prior; B1 metadata markers only; B2 heuristic signals
(the detector's estimate, which never uses markers). B3/B4 (with the LLM judge) come in
phase 4. Metrics are reported per repository and per file; files inherit their repository's
label, and repositories with a ``mixed`` label are left out of file-level metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_code_detector.aggregate import family_scores  # noqa: E402
from ai_code_detector.methodology import load_methodology  # noqa: E402
from ai_code_detector.models import SignalHit  # noqa: E402
from dataset import Sample, fetch, load_manifest  # noqa: E402
from margin_core import run_agent  # noqa: E402
from margin_core.llm import create_llm  # noqa: E402
from metrics import summarize  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7)
SECONDARY_FACTOR = load_methodology().secondary_factor


@dataclass(frozen=True)
class JudgeOptions:
    """When set, the detector also runs the LLM judge (baseline B3)."""

    provider: str
    model: str
    files: int = 8
    seconds: float = 900.0


async def evaluate_sample(
    sample: Sample,
    manifest_dir: Path,
    judge: JudgeOptions | None = None,
    methodology: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = fetch(sample, manifest_dir)
    params: dict[str, Any] = {"target": str(target), "use_llm_judge": False}
    llm = None
    if judge is not None:
        params.update(
            use_llm_judge=True, max_files_for_judge=judge.files, max_judge_seconds=judge.seconds
        )
        llm = create_llm(judge.provider, judge.model)
    config = {"methodology": methodology} if methodology else None
    result = await run_agent("ai-code-detector", params, llm=llm, config=config)
    record: dict[str, Any] = {
        "id": sample.id,
        "label": sample.label,
        "is_ai": sample.is_ai,
        "era": sample.era,
        "language": sample.language,
        "split": sample.split,
        "status": result.status,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if result.status == "error":
        record["error"] = result.error
        return record
    data = result.data
    record.update(
        estimate=data["estimate"],
        estimate_range=data["estimate_range"],
        confidence=data["confidence"],
        has_markers=bool(data["metadata_markers"]),
        marker_tools=sorted({m["tool"] for m in data["metadata_markers"]}),
        files_analyzed=data["files_analyzed"],
        warnings=result.warnings,
        llm_calls=result.metadata.usage.llm_calls,
        methodology_version=data["methodology_version"],
        files=[
            {
                "path": f["path"],
                "lines": f["lines"],
                "language": f["language"],
                "probability": f["probability"],
                "signals": {h["signal_id"]: h["strength"] for h in f["signals"]},
                "families": _families(f["signals"]),
                "judge": f["judge"]["likelihood_ai"] if f.get("judge") else None,
                "judge_confidence": f["judge"]["confidence"] if f.get("judge") else None,
            }
            for f in data["files"]
            if not f["is_test"]
        ],
        repo_signals={h["signal_id"]: h["strength"] for h in data["repo_signals"]},
        repo_families=_families(data["repo_signals"]),
    )
    return record


def _families(hits: list[dict[str, Any]]) -> dict[str, float]:
    """Family scores exactly as the detector combines them (strongest + reduced others)."""
    return family_scores([SignalHit.model_validate(h) for h in hits], SECONDARY_FACTOR)


FILE_CAP = 50  # files per repository in file-level metrics, so big repos cannot dominate


def _capped_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    files = sorted(record["files"], key=lambda f: hashlib.sha1(f["path"].encode()).hexdigest())
    return files[:FILE_CAP]


def _metrics_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r["status"] != "error"]
    pure = [r for r in ok if r["label"] != "mixed"]
    labels = [r["is_ai"] for r in ok]
    prior = sum(labels) / len(labels) if labels else 0.5
    repo = {
        "B0_constant_prior": summarize([prior] * len(ok), labels, THRESHOLDS),
        "B1_metadata_only": summarize(
            [1.0 if r["has_markers"] else 0.0 for r in ok], labels, THRESHOLDS
        ),
        "B2_signals": summarize([r["estimate"] for r in ok], labels, THRESHOLDS),
    }
    file_scores = [f["probability"] for r in pure for f in _capped_files(r)]
    file_labels = [r["is_ai"] for r in pure for _ in _capped_files(r)]
    by_group: dict[str, Any] = {}
    for key in ("era", "language"):
        for value in sorted({str(r.get(key)) for r in ok}):
            members = [r for r in ok if str(r.get(key)) == value]
            by_group[f"{key}={value}"] = {
                "n": len(members),
                "mean_estimate_human": _mean([r["estimate"] for r in members if not r["is_ai"]]),
                "mean_estimate_ai": _mean([r["estimate"] for r in members if r["is_ai"]]),
            }
    return {
        "repository_level": repo,
        "file_level": {"B2_signals": summarize(file_scores, file_labels, THRESHOLDS)},
        "groups": by_group,
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = sorted({r["split"] for r in records})
    result: dict[str, Any] = {split: _metrics_for([r for r in records if r["split"] == split])
                              for split in splits}  # fmt: skip
    if len(splits) > 1:
        result["combined"] = _metrics_for(records)
    result["errors"] = [r["id"] for r in records if r["status"] == "error"]
    return result


async def main_async(
    manifest: Path,
    out: Path,
    only: set[str] | None,
    splits: set[str] | None = None,
    judge: JudgeOptions | None = None,
    methodology: str | None = None,
) -> dict[str, Any]:
    samples = [
        s for s in load_manifest(manifest)
        if (not only or s.id in only) and (splits is None or s.split in splits)
    ]  # fmt: skip
    records = []
    for i, sample in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] {sample.id} ({sample.label})", file=sys.stderr, flush=True)
        try:
            records.append(await evaluate_sample(sample, manifest.parent, judge, methodology))
        except Exception as exc:  # a broken sample must not stop the evaluation
            records.append({"id": sample.id, "label": sample.label, "is_ai": sample.is_ai,
                            "era": sample.era, "split": sample.split, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})  # fmt: skip
        last = records[-1]
        if last["status"] != "error":
            summary = (
                f"estimate={last['estimate']:.3f} markers={last['marker_tools']} "
                f"files={last['files_analyzed']} ({last['seconds']}s)"
            )
            print(f"      {summary}", file=sys.stderr)
        else:
            print(f"      ERROR {last['error']}", file=sys.stderr)
    report = {
        "dataset": manifest.parent.name,
        "manifest": manifest.as_posix(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "methodology_version": next(
            (r["methodology_version"] for r in records if "methodology_version" in r), None
        ),
        "metrics": compute_metrics(records),
        "samples": records,
    }
    await asyncio.to_thread(_write_report, out, report)
    return report


def _write_report(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the AI Code Detector.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="evaluate only these sample ids")
    parser.add_argument("--methodology", help="methodology version (default: the detector's)")
    parser.add_argument("--judge-model", help="also run the LLM judge with this model (B3)")
    parser.add_argument("--judge-provider", default="local")
    parser.add_argument("--judge-files", type=int, default=8, help="files judged per repository")
    parser.add_argument(
        "--final",
        action="store_true",
        help="also score the test split (do this once per methodology version, never while tuning)",
    )
    args = parser.parse_args(argv)
    splits = {"train", "val", "test"} if args.final else {"train", "val"}
    if args.final:
        print("Scoring the TEST split. Do not tune anything on these numbers.", file=sys.stderr)
    judge = (
        JudgeOptions(args.judge_provider, args.judge_model, args.judge_files)
        if args.judge_model
        else None
    )
    only = set(args.only or []) or None
    report = asyncio.run(main_async(args.manifest, args.out, only, splits, judge, args.methodology))
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
