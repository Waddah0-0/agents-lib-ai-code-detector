"""Pin GitHub repositories to their last commit before a date (for pre-AI human samples).

Usage::

    uv run python agents/ai_code_detector/eval/tools/pin_before.py --before 2021-01-01T00:00:00Z \\
        pallets/itsdangerous hynek/structlog ...

Prints one JSON object per repository: the pinned commit, its date, and how many Python
files it has. Uses blobless clones; no GitHub API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=300,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()  # fmt: skip


def pin(full_name: str, before: str) -> dict[str, object]:
    workdir = Path(tempfile.mkdtemp(prefix="pin-"))
    try:
        url = f"https://github.com/{full_name}.git"
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(workdir / "r")],
            check=True, capture_output=True, timeout=300,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )  # fmt: skip
        repo = workdir / "r"
        sha = git(repo, "rev-list", "-1", "--first-parent", f"--before={before}", "HEAD")
        if not sha:
            return {"repo": full_name, "error": f"no commit before {before}"}
        files = git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        return {
            "repo": full_name,
            "commit": sha,
            "date": git(repo, "show", "-s", "--format=%cI", sha),
            "commits_until_then": int(git(repo, "rev-list", "--count", sha)),
            "py_files": sum(1 for f in files if f.endswith(".py")),
            "files": len(files),
        }
    except subprocess.SubprocessError as exc:
        return {"repo": full_name, "error": type(exc).__name__}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--before",
        required=True,
        help="exact timestamp, e.g. 2021-01-01T00:00:00Z (a bare date means local time now)",
    )
    parser.add_argument("repos", nargs="+", help="owner/name")
    args = parser.parse_args(argv)
    for name in args.repos:
        print(json.dumps(pin(name, args.before)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
