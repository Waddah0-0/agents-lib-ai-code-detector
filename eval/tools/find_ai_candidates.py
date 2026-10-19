"""Find public Python repositories whose history shows they were built with AI tools.

How a repository qualifies (all must hold):

- found through GitHub commit search for a verified AI-tool marker (markers/v1.toml);
- not a fork; at most ``--max-files`` files;
- at least ``--min-commits`` commits, and at least ``--min-marker-share`` of them carry an
  AI marker (trailer, generated-with line or bot identity);
- at least ``--min-files-of-language`` files in ``--language``, and that language makes up
  most of its code files;
- first commit on or after ``--since``.

Markers are used only to *select and label* samples; the detector's estimate never uses them,
so this does not leak the label into the score being evaluated.

Usage: ``uv run python agents/ai_code_detector/eval/tools/find_ai_candidates.py --out c.json``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_code_detector.markers import commit_marker_tools, load_markers
from margin_core.config import get_settings
from margin_core.loaders.git import read_commits_sync

QUERIES = (
    '"Generated with Claude Code"',
    '"noreply@anthropic.com"',
    '"cursoragent@cursor.com"',
    '"Co-authored-by: Codex"',
    "author-email:198982749+Copilot@users.noreply.github.com",
    "author:devin-ai-integration[bot]",
    "author:google-labs-jules[bot]",
    '"Co-Authored-By: Gemini"',
    'author-name:"(aider)"',
)
LANGUAGE_SUFFIXES = {
    "python": {".py"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"},
}
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php", ".cs",
                 ".cpp", ".c", ".kt", ".swift", ".vue", ".svelte"}  # fmt: skip


def search_repos(query: str, pages: int) -> list[str]:
    found: list[str] = []
    for page in range(1, pages + 1):
        url = "https://api.github.com/search/commits?per_page=100&sort=committer-date&page="
        url += f"{page}&q=" + urllib.parse.quote(query)
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "margin-eval"}
        secret = get_settings().github_token  # GITHUB_TOKEN env var or .env
        token = secret.get_secret_value() if secret else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            data = json.load(urllib.request.urlopen(request, timeout=30))
        except Exception as exc:
            print(f"  search failed ({exc}); waiting 60s", file=sys.stderr)
            time.sleep(60)
            continue
        for item in data.get("items", []):
            repo = item["repository"]
            if not repo.get("fork"):
                found.append(repo["full_name"])
        time.sleep(8)  # unauthenticated search allows about 10 requests per minute
    return list(dict.fromkeys(found))


def inspect(full_name: str, args: argparse.Namespace) -> dict[str, Any] | None:
    markers = load_markers()
    workdir = Path(tempfile.mkdtemp(prefix="cand-"))
    try:
        url = f"https://github.com/{full_name}.git"
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(workdir / "r")],
            check=True, capture_output=True, timeout=180,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )  # fmt: skip
        repo = workdir / "r"
        names = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.splitlines()  # fmt: skip
        if len(names) > args.max_files:
            return None
        code = [n for n in names if Path(n).suffix in CODE_SUFFIXES]
        py = [n for n in code if Path(n).suffix in LANGUAGE_SUFFIXES[args.language]]
        if len(py) < args.min_files_of_language or len(py) < 0.6 * len(code):
            return None
        commits = read_commits_sync(repo)
        if len(commits) < args.min_commits:
            return None
        marked = [c for c in commits if commit_marker_tools(c, markers)]
        share = len(marked) / len(commits)
        first = min(c.authored_at for c in commits)
        if share < args.min_marker_share or first < datetime.fromisoformat(args.since):
            return None
        tools = sorted({tool for c in marked for _, tool in commit_marker_tools(c, markers)})
        return {
            "repo": full_name,
            "head": commits[0].sha,
            "commits": len(commits),
            "marker_share": round(share, 3),
            "tools": tools,
            "language": args.language,
            "language_files": len(py),
            "code_files": len(code),
            "first_commit": first.isoformat(),
        }
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        print(f"  {full_name}: skipped ({type(exc).__name__})", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--min-commits", type=int, default=10)
    parser.add_argument("--min-marker-share", type=float, default=0.8)
    parser.add_argument("--language", choices=sorted(LANGUAGE_SUFFIXES), default="python")
    parser.add_argument("--min-files-of-language", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--since", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--per-query", type=int, default=4, help="qualifying repos per query")
    parser.add_argument(
        "--query", action="append", help="search only these queries (default: all markers)"
    )
    parser.add_argument("--exclude", type=Path, help="earlier candidates file to skip")
    args = parser.parse_args(argv)

    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    if args.exclude:
        earlier = json.loads(args.exclude.read_text(encoding="utf-8"))["candidates"]
        seen |= {c["repo"] for c in earlier}
    for query in args.query or QUERIES:
        print(f"query {query}", file=sys.stderr)
        kept = 0
        for name in search_repos(query, args.pages):
            if name in seen or kept >= args.per_query:
                continue
            seen.add(name)
            result = inspect(name, args)
            if result:
                kept += 1
                accepted.append({**result, "query": query})
                line = (
                    f"  + {name}: {result['commits']} commits, {result['marker_share']:.0%} "
                    f"marked, {result['language_files']} {args.language} files, {result['tools']}"
                )
                print(line, file=sys.stderr)
    report = {
        "created": datetime.now(UTC).isoformat(),
        "criteria": {k: v for k, v in vars(args).items() if k != "out"},
        "candidates": accepted,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{len(accepted)} candidates written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
