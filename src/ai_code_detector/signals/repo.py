"""Repository-level signals: commit history shape and README style."""

from __future__ import annotations

import re

from agents_core import Evidence

from ai_code_detector.models import SignalHit
from ai_code_detector.signals.base import RepoContext, RepoSignal, ramp

_CONVENTIONAL = re.compile(r"^[a-z]+(\([^)]*\))?!?: ", re.IGNORECASE)
_MESSY = re.compile(
    r"\b(typo|oops|whoops|wip|tmp|temp|again|stuff|asdf|minor|forgot|small fix|quick fix|"
    r"try|trying|maybe|hopefully|fixup|squash)\b|^revert\b|^merge\b",
    re.IGNORECASE,
)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
_TEMPLATE_SECTIONS = {
    "features", "installation", "usage", "contributing", "license", "getting started",
    "prerequisites", "project structure", "table of contents", "acknowledgments",
    "acknowledgements", "roadmap", "tech stack", "configuration", "requirements",
}  # fmt: skip
_MARKETING = re.compile(
    r"\b(seamless(ly)?|powerful|robust|comprehensive|cutting-edge|effortless(ly)?|"
    r"blazing(ly)?|leverag\w*|streamlin\w*|intuitive|elevate|empower\w*|delve|unlock|"
    r"revolutioni[sz]\w*|state-of-the-art|game-chang\w*)\b",
    re.IGNORECASE,
)
_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md", "Readme.md")


class CommitShape(RepoSignal):
    """A finished project arriving in one go, or many very large commits."""

    id = "commit_shape"
    version = "1"
    family = "history"

    def evaluate(self, ctx: RepoContext) -> list[SignalHit]:
        if not ctx.commits:
            return []
        t = ctx.methodology.thresholds_for(self.id)
        current = {f.path for f in ctx.workspace.files}
        if len(current) < t["min_files"]:
            return []

        if len(ctx.commits) == 1:
            sha = ctx.commits[0].sha
            return [
                self.hit(
                    t["single_commit_strength"],
                    "The whole project arrived in a single commit. AI-built projects often do, "
                    "but so do existing projects uploaded at once, so this is weak evidence.",
                    [Evidence(kind="commit", location=sha, detail="only commit")],
                )
            ]

        strength = 0.0
        evidence: list[Evidence] = []
        parts: list[str] = []
        roots = [c for c in ctx.commits if not c.parents]
        if roots and not ctx.history_truncated:
            root = roots[-1]
            share = len(set(ctx.changed_files.get(root.sha, [])) & current) / len(current)
            if share >= t["first_share"]:
                strength += 0.3 + 0.4 * ramp(share, t["first_share"], 1.0)
                parts.append(f"the first commit already contained {share:.0%} of today's files")
                evidence.append(
                    Evidence(kind="commit", location=root.sha, detail=f"first commit: {share:.0%}")
                )

        non_merge = [c for c in ctx.commits if len(c.parents) <= 1]
        if len(non_merge) >= t["min_commits_for_large"]:
            large = [
                c
                for c in non_merge
                if len(ctx.changed_files.get(c.sha, [])) >= t["large_commit_files"]
            ]
            share = len(large) / len(non_merge)
            if share >= t["large_commit_share"]:
                strength += 0.3
                parts.append(
                    f"{share:.0%} of commits each touched {int(t['large_commit_files'])}+ files"
                )
                evidence += [
                    Evidence(
                        kind="commit",
                        location=c.sha,
                        detail=f"{len(ctx.changed_files.get(c.sha, []))} files: {c.subject[:60]}",
                    )
                    for c in large[:3]
                ]
        if not parts:
            return []
        return [self.hit(strength, "History shape: " + "; ".join(parts) + ".", evidence)]


class HistoryDepth(RepoSignal):
    """A long, irregular history with small fixes and false starts suggests people at work."""

    id = "history_depth"
    version = "1"
    family = "history"

    @staticmethod
    def _messy(subject: str) -> bool:
        if _MESSY.search(subject):
            return True
        return not _CONVENTIONAL.match(subject) and len(subject.split()) <= 2

    def evaluate(self, ctx: RepoContext) -> list[SignalHit]:
        t = ctx.methodology.thresholds_for(self.id)
        commits = ctx.commits
        if len(commits) < t["small_min_commits"]:
            return []
        dates = [c.authored_at for c in commits]
        span_days = (max(dates) - min(dates)).days
        messy = [c for c in commits if self._messy(c.subject)]
        share = len(messy) / len(commits)
        if len(commits) >= t["min_commits"] and span_days >= t["min_span_days"]:
            if share < t["messy_share"]:
                return []
            strength = -(0.3 + 0.4 * ramp(share, t["messy_share"], 0.5))
        elif span_days >= t["small_min_span_days"] and share >= t["small_messy_share"]:
            strength = -0.3
        else:
            return []
        evidence = [
            Evidence(kind="commit", location=c.sha, detail=f"'{c.subject[:60]}'") for c in messy[:3]
        ]
        return [
            self.hit(
                strength,
                f"{len(commits)} commits over {span_days} days, {share:.0%} of them small fixes, "
                "reverts or terse messages: the irregular rhythm of people working.",
                evidence,
            )
        ]


class ReadmePatterns(RepoSignal):
    """Emoji headings, a template section list and marketing words are common in AI READMEs."""

    id = "readme_patterns"
    version = "1"
    family = "docs"

    def evaluate(self, ctx: RepoContext) -> list[SignalHit]:
        t = ctx.methodology.thresholds_for(self.id)
        readme = next((n for n in _README_NAMES if ctx.workspace.get_file(n)), None)
        if readme is None:
            return []
        text = ctx.workspace.read_text(readme, max_bytes=200_000)
        headings = [line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")]
        emoji = [h for h in headings if _EMOJI.search(h)]
        plain = {_EMOJI.sub("", h).strip(" :").lower() for h in headings}
        template = plain & _TEMPLATE_SECTIONS
        marketing = {m.group(0).lower() for m in _MARKETING.finditer(text)}

        strength = 0.0
        parts: list[str] = []
        if len(emoji) >= t["emoji_headings"]:
            strength += 0.3
            parts.append(f"{len(emoji)} headings start with emoji")
        if len(template) >= t["template_sections"]:
            strength += 0.2
            parts.append(f"a template section list ({', '.join(sorted(template))})")
        if len(marketing) >= t["marketing_words"]:
            strength += 0.2
            parts.append(f"marketing words ({', '.join(sorted(marketing)[:5])})")
        if not parts:
            return []
        return [
            self.hit(
                min(t["max_strength"], strength),
                "README style: " + "; ".join(parts) + ".",
                [Evidence(kind="file", location=readme, detail="; ".join(parts))],
            )
        ]


REPO_SIGNALS: tuple[RepoSignal, ...] = (CommitShape(), HistoryDepth(), ReadmePatterns())
