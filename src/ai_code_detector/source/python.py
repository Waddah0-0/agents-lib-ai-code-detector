"""Python parser, using the standard library (``ast`` + ``tokenize``)."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter

from ai_code_detector.source import DEBUG_WORDS, Comment, Function, SourceFile

_NON_CODE_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}
_CODE_HINT = re.compile(r"[(=\[]|^\s*(import|from|return|if|for|while|def|class)\b")


def _docstring_node(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[0].value
    return None


def looks_like_python(text: str) -> bool:
    """Whether a comment's text is commented-out code rather than prose."""
    if not _CODE_HINT.search(text) or text.endswith((".", "?", "!")):
        return False
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def parse_python(path: str, text: str) -> SourceFile | None:
    """Analyse ``text``; ``None`` if it is not valid Python."""
    try:
        tree = ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        return None

    docstring_line_numbers: set[int] = set()
    functions: list[Function] = []
    identifiers: list[str] = []
    debug: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        doc = _docstring_node(node) if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) else None  # fmt: skip
        if doc is not None and doc.end_lineno is not None:
            docstring_line_numbers.update(range(doc.lineno, doc.end_lineno + 1))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body[1:] if doc is not None else node.body
            functions.append(
                Function(node.name, node.lineno, node.end_lineno or node.lineno, len(body),
                         doc is not None)
            )  # fmt: skip
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
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )  # fmt: skip
            first = node.args[0] if node.args else None
            literal = first.value if isinstance(first, ast.Constant) else None
            if (
                name == "print"
                and isinstance(literal, str)
                and literal.strip(" .!:").lower() in DEBUG_WORDS
            ):
                debug.append((node.lineno, f'print("{literal}")'))
            elif name in {"breakpoint", "set_trace"}:
                debug.append((node.lineno, f"{name}()"))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [
                node.module or ""
            ]  # fmt: skip
            debug += [(node.lineno, f"import {m}") for m in modules if m in {"pdb", "ipdb"}]

    code_lines: set[int] = set()
    comments: list[Comment] = []
    quote_styles: Counter[str] = Counter()
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

    return SourceFile(
        path=path,
        language="python",
        lines=text.splitlines(),
        code_line_numbers=code_lines - docstring_line_numbers,
        comments=comments,
        docstring_lines=len(docstring_line_numbers),
        functions=functions,
        identifiers=identifiers,
        quote_styles=quote_styles,
        commented_code_lines=[c.line for c in comments if c.own_line and looks_like_python(c.text)],
        debug_statements=debug,
    )
