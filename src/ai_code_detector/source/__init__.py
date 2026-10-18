"""Language-neutral facts about one source file, produced by a parser per language.

Signals read only :class:`SourceFile`, so adding a language means adding a parser here, not
changing signals. Language-specific judgements (what counts as a docstring, which comments
are commented-out code, which calls are leftover debugging) are made by the parser.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
DEBUG_WORDS = {"here", "debug", "test", "xxx", "ok", "hello", "asdf", "got here", "reached"}


@dataclass(frozen=True)
class Comment:
    line: int
    text: str  # without comment markers, stripped (block comments joined with spaces)
    own_line: bool  # True when nothing but whitespace precedes it
    line_count: int = 1  # lines spanned (block comments)


@dataclass(frozen=True)
class Function:
    name: str
    line: int
    end_line: int
    statements: int  # body statements, not counting a docstring
    has_docstring: bool  # Python docstring, or a JSDoc block directly above


@dataclass
class SourceFile:
    path: str
    language: str
    lines: list[str]
    code_line_numbers: set[int]
    comments: list[Comment]
    docstring_lines: int
    functions: list[Function]
    identifiers: list[str]
    quote_styles: Counter[str] = field(default_factory=Counter)
    commented_code_lines: list[int] = field(default_factory=list)
    debug_statements: list[tuple[int, str]] = field(default_factory=list)

    @property
    def code_lines(self) -> int:
        return len(self.code_line_numbers)

    @property
    def own_line_comment_lines(self) -> int:
        return sum(c.line_count for c in self.comments if c.own_line)

    def next_code_line(self, after: int) -> int | None:
        later = [n for n in self.code_line_numbers if n > after]
        return min(later) if later else None


def split_words(text: str) -> list[str]:
    """Lower-case words from prose or identifiers (``getUserName`` -> get, user, name)."""
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        words += [w.lower() for w in _WORD_RE.findall(chunk)]
    return words


PYTHON_SUFFIXES = {".py"}
JAVASCRIPT_SUFFIXES = {".js", ".jsx", ".mjs", ".cjs"}
TYPESCRIPT_SUFFIXES = {".ts", ".mts", ".cts", ".tsx"}
SUPPORTED_SUFFIXES = PYTHON_SUFFIXES | JAVASCRIPT_SUFFIXES | TYPESCRIPT_SUFFIXES


def is_supported(path: str) -> bool:
    lower = path.lower()
    if lower.endswith((".d.ts", ".d.mts", ".d.cts")):  # type declarations, not code
        return False
    return PurePosixPath(lower).suffix in SUPPORTED_SUFFIXES


def parse_source(path: str, text: str) -> SourceFile | None:
    """Parse ``text`` with the parser for ``path``'s language; ``None`` if it cannot."""
    suffix = PurePosixPath(path.lower()).suffix
    if suffix in PYTHON_SUFFIXES:
        from ai_code_detector.source.python import parse_python

        return parse_python(path, text)
    if suffix in JAVASCRIPT_SUFFIXES | TYPESCRIPT_SUFFIXES:
        from ai_code_detector.source.javascript import parse_javascript

        return parse_javascript(path, text)
    return None
