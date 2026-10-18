"""Choose which files the LLM judge reads.

Small files, tests and very low-information files are skipped. The rest are split into thirds
by their heuristic probability (most AI-like, most human-like, in between) and picked in
turn, so the judge also sees likely-human code and is not only asked to confirm the
heuristics. The order is seeded, so the same repository always gives the same sample.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from ai_code_detector.models import FileScore


def sample_files(
    files: Sequence[FileScore],
    *,
    max_files: int,
    min_lines: int,
    seed_text: str,
) -> list[FileScore]:
    eligible = [f for f in files if f.lines >= min_lines and not f.is_test]
    if len(eligible) <= max_files:
        return sorted(eligible, key=lambda f: f.path)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    ordered = sorted(eligible, key=lambda f: (f.probability, f.path))
    third = len(ordered) // 3
    groups = [ordered[2 * third :], ordered[:third], ordered[third : 2 * third]]
    for group in groups:
        rng.shuffle(group)
    picked: list[FileScore] = []
    while len(picked) < max_files and any(groups):
        for group in groups:
            if group and len(picked) < max_files:
                picked.append(group.pop())
    return sorted(picked, key=lambda f: f.path)
