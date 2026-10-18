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
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents_core import run_agent  # noqa: E402
from dataset import Sample, fetch, load_manifest  # noqa: E402
from metrics import summarize  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7)


async def evaluate_sample(sample: Sample, manifest_dir: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = fetch(sample, manifest_dir)
    result = await run_agent("ai-code-detector", {"target": str(target)})
    record: dict[str, Any] = {
        "id": sample.id,
        "label": sample.label,
        "is_ai": sample.is_ai,
        "era": sample.era,
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
        methodology_version=data["methodology_version"],
        files=[{"path": f["path"], "probability": f["probability"]} for f in data["files"]],
        repo_signals={h["signal_id"]: h["strength"] for h in data["repo_signals"]},
    )
    return record


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    file_scores = [f["probability"] for r in pure for f in r["files"]]
    file_labels = [r["is_ai"] for r in pure for _ in r["files"]]
    by_era: dict[str, Any] = {}
    for era in sorted({r["era"] for r in ok}):
        members = [r for r in ok if r["era"] == era]
        by_era[era] = {
            "n": len(members),
            "mean_estimate": round(sum(r["estimate"] for r in members) / len(members), 4),
        }
    return {
        "repository_level": repo,
        "file_level": {"B2_signals": summarize(file_scores, file_labels, THRESHOLDS)},
        "by_era": by_era,
        "errors": [r["id"] for r in records if r["status"] == "error"],
    }


async def main_async(manifest: Path, out: Path, only: set[str] | None) -> dict[str, Any]:
    samples = [s for s in load_manifest(manifest) if not only or s.id in only]
    records = []
    for i, sample in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] {sample.id} ({sample.label})", file=sys.stderr, flush=True)
        try:
            records.append(await evaluate_sample(sample, manifest.parent))
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
    args = parser.parse_args(argv)
    report = asyncio.run(main_async(args.manifest, args.out, set(args.only or []) or None))
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
