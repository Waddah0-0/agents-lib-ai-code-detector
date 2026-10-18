"""Facts about one Python file, extracted with the standard library (``ast`` + ``tokenize``)."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field

_NON_CODE_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@dataclass(frozen=True)
class Comment:
    line: int
    text: str  # without the leading "#", stripped
    own_line: bool  # True when nothing but whitespace precedes it


@dataclass(frozen=True)
class Function:
    name: str
    line: int
    end_line: int
    statements: int  # body statements, not counting the docstring
    has_docstring: bool


@dataclass
class PythonFile:
    path: str
    lines: list[str]
    code_line_numbers: set[int]
    comments: list[Comment]
    docstring_lines: int
    functions: list[Function]
    identifiers: list[str]
    quote_styles: Counter[str] = field(default_factory=Counter)
    string_literals: list[tuple[int, str]] = field(default_factory=list)
    calls: list[tuple[int, str, list[str]]] = field(default_factory=list)  # (line, name, str args)
    imports: list[tuple[int, str]] = field(default_factory=list)

    @property
    def code_lines(self) -> int:
        return len(self.code_line_numbers)

    def next_code_line(self, after: int) -> int | None:
        later = [n for n in self.code_line_numbers if n > after]
        return min(later) if later else None


def split_words(text: str) -> list[str]:
    """Lower-case words from prose or identifiers (``getUserName`` -> get, user, name)."""
    words: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        words += [w.lower() for w in _WORD_RE.findall(chunk)]
    return words


def _docstring_node(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[0].value
    return None


def parse_python(path: str, text: str) -> PythonFile | None:
    """Analyse ``text``; ``None`` if it is not valid Python."""
    try:
        tree = ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        return None

    lines = text.splitlines()
    docstring_line_numbers: set[int] = set()
    functions: list[Function] = []
    identifiers: list[str] = []
    calls: list[tuple[int, str, list[str]]] = []
    imports: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        doc = _docstring_node(node) if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) else None  # fmt: skip
        if doc is not None and doc.end_lineno is not None:
            docstring_line_numbers.update(range(doc.lineno, doc.end_lineno + 1))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body[1:] if doc is not None else node.body
            functions.append(
                Function(
                    name=node.name,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    statements=len(body),
                    has_docstring=doc is not None,
                )
            )
            identifiers.append(node.name)
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if arg.arg not in {"self", "cls"}:
                    identifiers.append(arg.arg)
        elif isinstance(node, ast.ClassDef):
            identifiers.append(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            identifiers.append(node.id)
        elif isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            str_args = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            calls.append((node.lineno, name, str_args))
        elif isinstance(node, ast.Import):
            imports += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))

    code_lines: set[int] = set()
    comments: list[Comment] = []
    quote_styles: Counter[str] = Counter()
    string_literals: list[tuple[int, str]] = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            before = tok.line[: tok.start[1]]
            comments.append(
                Comment(tok.start[0], tok.string.lstrip("#").strip(), not before.strip())
            )
        elif tok.type not in _NON_CODE_TOKENS:
            code_lines.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type == tokenize.STRING:
            literal = tok.string.lstrip("rRbBuUfF")
            if (
                not literal.startswith(('"""', "'''"))
                and tok.start[0] not in docstring_line_numbers
            ):
                quote_styles[literal[0]] += 1
                string_literals.append((tok.start[0], literal[1:-1]))

    return PythonFile(
        path=path,
        lines=lines,
        code_line_numbers=code_lines - docstring_line_numbers,
        comments=comments,
        docstring_lines=len(docstring_line_numbers),
        functions=functions,
        identifiers=identifiers,
        quote_styles=quote_styles,
        string_literals=string_literals,
        calls=calls,
        imports=imports,
    )
