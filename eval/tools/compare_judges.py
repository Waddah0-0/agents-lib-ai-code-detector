"""Quick head-to-head of judge models on the same files from TRAIN-split repositories.

Usage::

    uv run python agents/ai_code_detector/eval/tools/compare_judges.py \\
        --manifest agents/ai_code_detector/eval/datasets/v1/manifest.jsonl \\
        --models qwen2.5:7b qwen3.5:9b --files-per-repo 2 --out comparison.json

For each model: success rate, seconds per file, mean likelihood on AI and on human files,
file-level AUC, and the mean on recent (post-2023) human files, the hardest case. Uses the
evaluation cache, so repositories must have been fetched by eval/run.py before. Never reads
validation or test repositories.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ai_code_detector.judge.llm_judge import Judge, JudgeSettings  # noqa: E402
from ai_code_detector.methodology import load_methodology  # noqa: E402
from ai_code_detector.source import is_supported  # noqa: E402
from dataset import Sample, fetch, load_manifest  # noqa: E402
from margin_core.llm import create_llm  # noqa: E402
from metrics import auc  # noqa: E402


def pick_files(sample: Sample, root: Path, count: int) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or ".git/" in rel
            or "node_modules/" in rel
            or "test" in rel.lower()
            or not is_supported(rel)
        ):
            continue  # fmt: skip
        try:
            lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if 60 <= lines <= 400:
            candidates.append(path)
    candidates.sort(key=lambda p: hashlib.sha1(f"{sample.id}/{p.name}".encode()).hexdigest())
    return candidates[:count]


async def judge_all(model: str, items: list[tuple[Sample, Path, Path]]) -> dict[str, Any]:
    cfg = load_methodology().section("judge")
    settings = JudgeSettings(
        window_lines=int(cfg["window_lines"]),
        token_budget=int(cfg["token_budget"]),
        max_output_tokens=int(cfg["max_output_tokens"]),
    )
    rows: list[dict[str, Any]] = []
    for sample, root, path in items:
        llm = create_llm("local", model).session()
        t0 = time.perf_counter()
        try:
            verdict, _, warning = await Judge(llm, settings, cache=False).judge_file(
                path.relative_to(root).as_posix(), path.read_text("utf-8", errors="replace"), []
            )
            outcome: dict[str, Any] = {"likelihood": verdict.likelihood_ai if verdict else None,
                       "error": warning if verdict is None else None}  # fmt: skip
        except Exception as exc:
            outcome = {"likelihood": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
        row: dict[str, Any] = {
            "sample": sample.id,
            "is_ai": sample.is_ai,
            "era": sample.era,
            "file": path.relative_to(root).as_posix(),
            "seconds": round(time.perf_counter() - t0, 1),
            "output_tokens": llm.usage.output_tokens,
            **outcome,
        }
        rows.append(row)
        mark = f"{outcome['likelihood']:.2f}" if outcome["likelihood"] is not None else "FAIL"
        print(f"  {model:32} {sample.id:28} {rows[-1]['seconds']:5.1f}s {mark}", file=sys.stderr)
    ok = [r for r in rows if r["likelihood"] is not None]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "model": model,
        "success_rate": round(len(ok) / len(rows), 3),
        "seconds_per_file": mean([r["seconds"] for r in rows]),
        "mean_ai": mean([r["likelihood"] for r in ok if r["is_ai"]]),
        "mean_human": mean([r["likelihood"] for r in ok if not r["is_ai"]]),
        "mean_recent_human": mean(
            [r["likelihood"] for r in ok if not r["is_ai"] and r["era"] == "post2023"]
        ),
        "file_auc": auc([r["likelihood"] for r in ok], [r["is_ai"] for r in ok]),
        "rows": rows,
    }


async def main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = [s for s in load_manifest(args.manifest) if s.split == "train"]
    by_stratum: dict[tuple[bool, str, str], list[Sample]] = {}
    for s in sorted(samples, key=lambda s: s.id):
        by_stratum.setdefault((s.is_ai, s.language, s.era), []).append(s)
    chosen = [s for group in by_stratum.values() for s in group[: args.repos_per_stratum]]
    items = []
    for sample in chosen:
        root = fetch(sample, args.manifest.parent)
        items += [(sample, root, p) for p in pick_files(sample, root, args.files_per_repo)]
    print(f"{len(items)} files from {len(chosen)} train repositories", file=sys.stderr)
    return [await judge_all(model, items) for model in args.models]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare judge models on train files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--files-per-repo", type=int, default=2)
    parser.add_argument("--repos-per-stratum", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = asyncio.run(main_async(args))
    summary = [{k: v for k, v in r.items() if k != "rows"} for r in results]
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
