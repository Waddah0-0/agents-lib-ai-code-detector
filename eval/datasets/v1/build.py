"""Build manifest.jsonl for dataset v1 from the selections in sources/ and dataset v0.

Usage: ``uv run python agents/ai_code_detector/eval/datasets/v1/build.py``

Splits are assigned per repository (never per file), stratified by (label, language, era),
with a fixed seed: in each stratum 60% train, 20% validation, 20% test (rounded so that every
stratum with at least 3 repositories has one in validation and one in test).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
V0 = HERE.parent / "v0" / "manifest.jsonl"
SEED = 20261018
GENTOO_POLICY = (
    "Gentoo-ecosystem project; Gentoo's council banned AI-assisted contributions on 2024-04-14 "
    "(wiki.gentoo.org/wiki/Project:Council/AI_policy). Policy shows intent, not proof"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _sample_id(prefix: str, repo: str) -> str:
    return f"{prefix}-{repo.split('/')[1].lower()}"


def collect() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in V0.read_text("utf-8").splitlines():  # v0 was never used for tuning
        if line.strip() and not line.startswith("//"):
            samples.append({**json.loads(line), "language": "python"})
    for pin in _jsonl(SOURCES / "human_python_pre2021.jsonl"):
        samples.append(_human(pin, "python", "pre2021"))
    for pin in _jsonl(SOURCES / "human_javascript_pre2021.jsonl"):
        samples.append(_human(pin, "javascript", "pre2021"))
    for pin in _jsonl(SOURCES / "human_python_gentoo_post2023.jsonl"):
        samples.append(
            {**_human(pin, "python", "post2023"), "provenance": GENTOO_POLICY,
             "provenance_confidence": "medium"}
        )  # fmt: skip
    for name, language in (("ai_python.json", "python"), ("ai_javascript.json", "javascript")):
        for c in json.loads((SOURCES / name).read_text("utf-8"))["candidates"]:
            samples.append(
                {
                    "id": _sample_id("a", c["repo"]),
                    "label": "ai",
                    "era": "post2023",
                    "language": language,
                    "tool": ", ".join(c["tools"]),
                    "source": {"kind": "git", "url": f"https://github.com/{c['repo']}",
                               "commit": c["head"]},
                    "provenance": f"{c['marker_share']:.0%} of {c['commits']} commits carry "
                    f"AI-tool markers; first commit {c['first_commit'][:10]}",
                    "provenance_confidence": "medium",
                }
            )  # fmt: skip
    ids = [s["id"] for s in samples]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(f"duplicate sample ids: {sorted(duplicates)}")
    return samples


def _human(pin: dict[str, Any], language: str, era: str) -> dict[str, Any]:
    if "error" in pin:
        raise SystemExit(f"{pin['repo']}: {pin['error']}")
    return {
        "id": _sample_id("h", pin["repo"]),
        "label": "human",
        "era": era,
        "language": language,
        "source": {"kind": "git", "url": f"https://github.com/{pin['repo']}",
                   "commit": pin["commit"]},
        "provenance": f"last commit before 2021-01-01 ({pin['date'][:10]}); predates AI coding "
        "assistants",
        "provenance_confidence": "high",
    }  # fmt: skip


def assign_splits(samples: list[dict[str, Any]]) -> None:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        strata[(s["label"], s["language"], s["era"])].append(s)
    rng = random.Random(SEED)
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda s: s["id"])
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, round(0.2 * n)) if n >= 3 else 0
        n_val = max(1, round(0.2 * n)) if n >= 3 else 0
        for i, s in enumerate(members):
            s["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"


def main() -> None:
    samples = collect()
    assign_splits(samples)
    samples.sort(key=lambda s: (s["split"], s["label"], s["language"], s["id"]))
    lines = ["// Dataset v1 (phase 4). Built by build.py; see README.md. Do not edit by hand."]
    lines += [json.dumps(s, ensure_ascii=False) for s in samples]
    (HERE / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for s in samples:
        table[(s["split"], s["label"], s["language"], s["era"])] += 1
    for key in sorted(table):
        print(*key, table[key])
    print("total", len(samples))


if __name__ == "__main__":
    main()
