"""File-level signals for Python source."""

from __future__ import annotations

import ast
import re

from agents_core import Evidence

from ai_code_detector.methodology import Methodology
from ai_code_detector.models import SignalHit
from ai_code_detector.python_source import PythonFile, split_words
from ai_code_detector.signals.base import FileSignal, ramp

_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "with", "is", "are", "be",
    "this", "that", "it", "its", "we", "if", "then", "else", "from", "by", "as", "at", "into",
    "all", "each", "our", "now", "here", "will", "should", "can", "using", "use",
}  # fmt: skip
_GENERIC_NAMES = {
    "data", "result", "results", "output", "value", "values", "item", "items", "info", "response",
    "temp", "obj", "element", "elements", "input_data", "output_data", "processed_data",
    "result_list", "final_result", "user_input", "my_list", "my_dict", "data_list", "data_dict",
    "result_dict", "output_list", "input_value", "return_value", "new_list", "filtered_data",
}  # fmt: skip
_GENERIC_FUNCTION = re.compile(r"^(process|handle|perform|execute|manage)_")
_SHORT_OK = {"i", "j", "k", "x", "y", "z", "_", "n", "e", "f"}
_TODO = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
_TODO_PERSONAL = re.compile(r"\b(TODO|FIXME|XXX|HACK)\s*\(\s*\w+|\b(19|20)\d\d\b|\b[A-Z]{2,3}:")
_INFORMAL = re.compile(
    r"\b(hacky?|ugly|lol|wtf|dunno|kinda|gonna|idk|meh|yikes|crap|damn|oops|sorry|"
    r"magic number|no idea|why does this work|don't ask)\b",
    re.IGNORECASE,
)
_DEBUG_WORDS = {"here", "debug", "test", "xxx", "ok", "hello", "asdf", "got here", "reached"}
_CODE_HINT = re.compile(r"[(=\[]|^\s*(import|from|return|if|for|while|def|class)\b")


def _location(source: PythonFile, line: int, end: int | None = None) -> str:
    return f"{source.path}:{line}" + (f"-{end}" if end and end != line else "")


class CommentDensity(FileSignal):
    """Many comments and docstrings, especially on trivial functions, are typical of AI code."""

    id = "comment_density"
    version = "1"
    family = "comments"

    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]:
        t = methodology.thresholds_for(self.id)
        own_comments = sum(1 for c in source.comments if c.own_line)
        density = (own_comments + source.docstring_lines) / max(source.code_lines, 1)
        trivial = [f for f in source.functions if f.statements <= t["trivial_max_statements"]]
        documented = [f for f in trivial if f.has_docstring]

        density_part = ramp(density, t["density_low"], t["density_high"])
        trivial_part = 0.0
        ratio = len(documented) / len(trivial) if trivial else 0.0
        if len(trivial) >= t["trivial_min_count"]:
            trivial_part = ramp(ratio, t["trivial_ratio_low"], 1.0)

        if density_part or trivial_part:
            strength = min(1.0, 0.6 * density_part + 0.6 * trivial_part)
            parts = []
            if density_part:
                parts.append(f"{density:.0%} comment and docstring lines per code line")
            if trivial_part:
                parts.append(
                    f"{len(documented)} of {len(trivial)} trivial functions have docstrings"
                )
            evidence = [
                Evidence(kind="file", location=_location(source, f.line, f.end_line),
                         detail=f"docstring on trivial function '{f.name}'")
                for f in documented[:3]
            ] or [Evidence(kind="file", location=source.path, detail=parts[0])]  # fmt: skip
            return [self.hit(strength, "Heavy commenting: " + "; ".join(parts) + ".", evidence)]

        if density < t["sparse_density"] and source.code_lines >= t["sparse_min_code_lines"]:
            return [
                self.hit(
                    t["sparse_strength"],
                    f"Almost no comments ({own_comments} for {source.code_lines} code lines); "
                    "AI-written code is rarely this sparse.",
                    [Evidence(kind="file", location=source.path, detail="sparse comments")],
                )
            ]
        return []


class RestatingComments(FileSignal):
    """Comments that only repeat the code next to them (``# increment counter``)."""

    id = "restating_comments"
    version = "1"
    family = "comments"

    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]:
        t = methodology.thresholds_for(self.id)
        eligible = 0
        restating: list[tuple[int, str]] = []
        for comment in source.comments:
            words = [w for w in split_words(comment.text) if w not in _STOPWORDS and len(w) > 1]
            if not (t["min_words"] <= len(words) <= t["max_words"]):
                continue
            if _TODO.search(comment.text):
                continue
            target = comment.line if not comment.own_line else source.next_code_line(comment.line)
            if target is None or target > len(source.lines):
                continue
            code = source.lines[target - 1]
            if not comment.own_line:
                code = code.split("#", 1)[0]
            code_words = set(split_words(code))
            if not code_words:
                continue
            eligible += 1
            overlap = sum(1 for w in words if w in code_words or w.rstrip("s") in code_words)
            if overlap / len(words) >= t["min_overlap"]:
                restating.append((comment.line, comment.text))

        if len(restating) < t["min_count"] or not eligible:
            return []
        ratio = len(restating) / eligible
        if ratio < t["min_ratio"]:
            return []
        return [
            self.hit(
                min(1.0, 0.3 + ratio),
                f"{len(restating)} of {eligible} comments only restate the code next to them.",
                [
                    Evidence(kind="file", location=_location(source, line), detail=f"# {text[:80]}")
                    for line, text in restating[:3]
                ],
            )
        ]


class NamingPatterns(FileSignal):
    """Long, generic, uniformly descriptive names lean AI; terse abbreviations lean human."""

    id = "naming_patterns"
    version = "1"
    family = "naming"

    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]:
        t = methodology.thresholds_for(self.id)
        names = {n for n in source.identifiers if not n.startswith("__")}
        if len(names) < t["min_identifiers"]:
            return []
        generic = sorted(n for n in names if n in _GENERIC_NAMES or _GENERIC_FUNCTION.match(n))
        considered = [n for n in names if n not in _SHORT_OK]
        if not considered:
            return []
        avg_len = sum(len(n) for n in considered) / len(considered)
        short = [n for n in considered if len(n.strip("_")) <= 2]
        generic_ratio = len(generic) / len(names)
        short_ratio = len(short) / len(considered)

        if generic_ratio >= t["generic_ratio"] and avg_len >= t["min_avg_length"]:
            return [
                self.hit(
                    0.3 + 0.3 * ramp(generic_ratio, t["generic_ratio"], 0.4),
                    f"Generic descriptive names ({', '.join(generic[:5])}) and long identifiers "
                    f"(average {avg_len:.1f} characters).",
                    [Evidence(kind="file", location=source.path, detail=", ".join(generic[:8]))],
                )
            ]
        if short_ratio >= t["short_ratio"]:
            return [
                self.hit(
                    -0.3,
                    f"Terse abbreviated names ({', '.join(sorted(short)[:5])}), "
                    "a habit of people more than of AI tools.",
                    [
                        Evidence(
                            kind="file", location=source.path, detail=", ".join(sorted(short)[:8])
                        )
                    ],
                )
            ]
        return []


class Idiosyncrasy(FileSignal):
    """Personal TODOs, informal remarks and inconsistent style: human fingerprints."""

    id = "idiosyncrasy"
    version = "1"
    family = "human_traces"

    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]:
        t = methodology.thresholds_for(self.id)
        found: list[tuple[int, str]] = []
        for comment in source.comments:
            if _TODO.search(comment.text) and _TODO_PERSONAL.search(comment.text):
                found.append((comment.line, f"personal note: # {comment.text[:70]}"))
            elif _INFORMAL.search(comment.text):
                found.append((comment.line, f"informal remark: # {comment.text[:70]}"))
        strength = t["per_marker"] * len(found)
        parts = [f"{len(found)} personal or informal comments"] if found else []

        singles, doubles = source.quote_styles.get("'", 0), source.quote_styles.get('"', 0)
        total = singles + doubles
        if (
            total >= t["quote_min_strings"]
            and min(singles, doubles) / total >= t["quote_min_minority"]
        ):
            strength += t["quote_strength"]
            parts.append(f"mixed quote styles ({singles} single, {doubles} double)")
        if not parts:
            return []
        evidence = [
            Evidence(kind="file", location=_location(source, line), detail=text)
            for line, text in found[:3]
        ] or [Evidence(kind="file", location=source.path, detail=parts[-1])]
        return [
            self.hit(
                -min(t["max_strength"], strength),
                "Human fingerprints: " + "; ".join(parts) + ".",
                evidence,
            )
        ]


def _looks_like_code(text: str) -> bool:
    if not _CODE_HINT.search(text) or text.endswith((".", "?", "!")):
        return False
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


class DeadOrDebugCode(FileSignal):
    """Commented-out code and leftover debugging are rare in freshly generated code."""

    id = "dead_or_debug_code"
    version = "1"
    family = "human_traces"

    def evaluate(self, source: PythonFile, methodology: Methodology) -> list[SignalHit]:
        t = methodology.thresholds_for(self.id)
        blocks: list[int] = []
        previous = -2
        for comment in source.comments:
            if comment.own_line and _looks_like_code(comment.text):
                if comment.line != previous + 1:
                    blocks.append(comment.line)
                previous = comment.line

        debug: list[tuple[int, str]] = []
        for line, name, args in source.calls:
            if name == "print" and args and args[0].strip(" .!:").lower() in _DEBUG_WORDS:
                debug.append((line, f'print("{args[0]}")'))
            elif name in {"breakpoint", "set_trace"}:
                debug.append((line, f"{name}()"))
        debug += [(line, f"import {mod}") for line, mod in source.imports if mod in {"pdb", "ipdb"}]

        if not blocks and not debug:
            return []
        strength = t["per_block"] * len(blocks) + t["per_debug"] * len(debug)
        parts = []
        if blocks:
            parts.append(f"{len(blocks)} blocks of commented-out code")
        if debug:
            parts.append(f"{len(debug)} leftover debugging statements")
        evidence = [
            Evidence(kind="file", location=_location(source, line), detail="commented-out code")
            for line in blocks[:2]
        ] + [Evidence(kind="file", location=_location(source, line), detail=text)
             for line, text in debug[:2]]  # fmt: skip
        return [
            self.hit(
                -min(t["max_strength"], strength),
                "Work-in-progress traces: " + "; ".join(parts) + ".",
                evidence,
            )
        ]


FILE_SIGNALS: tuple[FileSignal, ...] = (
    CommentDensity(),
    RestatingComments(),
    NamingPatterns(),
    Idiosyncrasy(),
    DeadOrDebugCode(),
)
