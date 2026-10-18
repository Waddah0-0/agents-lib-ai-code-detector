"""JavaScript and TypeScript parser, using tree-sitter.

``.js/.jsx/.mjs/.cjs`` use the JavaScript grammar (which includes JSX); ``.ts/.mts/.cts``
the TypeScript grammar and ``.tsx`` the TSX grammar. A JSDoc block (``/** ... */``) counts as
a docstring, and documents a function when it ends on the line just above it.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from pathlib import PurePosixPath

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from ai_code_detector.source import DEBUG_WORDS, Comment, Function, SourceFile

MAX_ERROR_SHARE = 0.10  # more of the file than this inside ERROR nodes: treat as unparseable
_FUNCTIONS = {
    "function_declaration", "function_expression", "arrow_function", "method_definition",
    "generator_function_declaration", "generator_function",
}  # fmt: skip
_DECLARATION_WRAPPERS = {
    "lexical_declaration", "variable_declaration", "export_statement", "variable_declarator",
    "pair", "public_field_definition", "assignment_expression", "expression_statement",
}  # fmt: skip
_CODE_HINT = re.compile(
    r"[(=;{\[]|\b(const|let|var|return|if|for|while|function|import|export|await)\b"
)
_CONSOLE = {"log", "debug", "info", "warn", "error", "trace"}


@cache
def _parser(kind: str) -> Parser:
    language = {
        "javascript": lambda: tree_sitter_javascript.language(),
        "typescript": lambda: tree_sitter_typescript.language_typescript(),
        "tsx": lambda: tree_sitter_typescript.language_tsx(),
    }[kind]()
    return Parser(Language(language))


def _grammar(path: str) -> str:
    suffix = PurePosixPath(path.lower()).suffix
    if suffix == ".tsx":
        return "tsx"
    if suffix in {".ts", ".mts", ".cts"}:
        return "typescript"
    return "javascript"


def _text(data: bytes, node: Node) -> str:
    """Source text of ``node``, sliced from the bytes we parsed (cheaper than ``Node.text``)."""
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Node) -> list[Node]:
    out, stack = [], [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(current.children))
    return out


def _comment_text(raw: str) -> str:
    if raw.startswith("//"):
        return raw[2:].strip()
    body = raw[2:-2] if raw.endswith("*/") else raw[2:]
    lines = [line.strip().lstrip("*").strip() for line in body.splitlines()]
    return " ".join(line for line in lines if line)


def looks_like_javascript(text: str, kind: str) -> bool:
    """Whether a line comment's text is commented-out code rather than prose."""
    if not _CODE_HINT.search(text) or text.endswith((".", "?", "!", ":")):
        return False
    tree = _parser(kind).parse(text.encode("utf-8"))
    return not tree.root_node.has_error and tree.root_node.named_child_count > 0


def _function_name(data: bytes, node: Node) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(data, name)
    parent = node.parent
    if parent is not None and parent.type == "variable_declarator":
        target = parent.child_by_field_name("name")
        if target is not None:
            return _text(data, target)
    return "<anonymous>"


def _declaration_start(node: Node) -> int:
    """Row where the statement declaring ``node`` starts (JSDoc sits right above it)."""
    current = node
    while current.parent is not None and current.parent.type in _DECLARATION_WRAPPERS:
        current = current.parent
    return current.start_point.row


def _statements(node: Node) -> int:
    body = node.child_by_field_name("body")
    if body is None:
        return 0
    if body.type == "statement_block" or body.type == "class_body":
        return sum(1 for c in body.named_children if c.type != "comment")
    return 1  # concise arrow body: one expression


def parse_javascript(path: str, text: str) -> SourceFile | None:
    kind = _grammar(path)
    data = text.encode("utf-8")
    tree = _parser(kind).parse(data)
    nodes = _walk(tree.root_node)
    lines = text.splitlines()
    if tree.root_node.has_error:
        error_rows: set[int] = set()
        for node in nodes:
            if node.type == "ERROR" or node.is_missing:
                error_rows.update(range(node.start_point.row, node.end_point.row + 1))
        if len(error_rows) > MAX_ERROR_SHARE * max(len(lines), 1):
            return None

    comments: list[Comment] = []
    docstring_rows: set[int] = set()
    jsdoc_end_rows: set[int] = set()
    code_rows: set[int] = set()
    quote_styles: Counter[str] = Counter()
    identifiers: list[str] = []
    debug: list[tuple[int, str]] = []
    function_nodes: list[Node] = []

    for node in nodes:
        kind_name = node.type
        if kind_name == "comment":
            raw = _text(data, node)
            start, end = node.start_point.row, node.end_point.row
            if raw.startswith("/**"):
                docstring_rows.update(range(start, end + 1))
                jsdoc_end_rows.add(end)
                continue
            before = lines[start][: node.start_point.column] if start < len(lines) else ""
            comments.append(
                Comment(start + 1, _comment_text(raw), not before.strip(), end - start + 1)
            )
            continue
        if node.child_count == 0 and kind_name not in {"comment", "ERROR"}:
            code_rows.update(range(node.start_point.row, node.end_point.row + 1))
        if kind_name == "string" and _text(data, node)[:1] in {"'", '"'}:
            quote_styles[_text(data, node)[0]] += 1
        elif kind_name in _FUNCTIONS:
            function_nodes.append(node)
        elif kind_name == "debugger_statement":
            debug.append((node.start_point.row + 1, "debugger"))
        elif kind_name == "call_expression":
            callee = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if callee is not None and callee.type == "member_expression" and args is not None:
                obj, prop = callee.child_by_field_name("object"), callee.child_by_field_name(
                    "property"
                )  # fmt: skip
                first = args.named_children[0] if args.named_children else None
                if (
                    obj is not None
                    and prop is not None
                    and _text(data, obj) == "console"
                    and _text(data, prop) in _CONSOLE
                    and first is not None
                    and first.type == "string"
                    and _text(data, first)[1:-1].strip(" .!:").lower() in DEBUG_WORDS
                ):
                    debug.append((node.start_point.row + 1, _text(data, node)[:60]))  # fmt: skip
        if kind_name == "variable_declarator":
            name = node.child_by_field_name("name")
            if name is not None and name.type == "identifier":
                identifiers.append(_text(data, name))
        elif kind_name in {"required_parameter", "optional_parameter"}:
            pattern = node.child_by_field_name("pattern")
            if pattern is not None and pattern.type == "identifier":
                identifiers.append(_text(data, pattern))
        elif kind_name == "formal_parameters":
            identifiers += [_text(data, c) for c in node.named_children if c.type == "identifier"]
        elif kind_name in {"class_declaration", "class"}:
            name = node.child_by_field_name("name")
            if name is not None:
                identifiers.append(_text(data, name))

    functions = []
    for node in function_nodes:
        function_name = _function_name(data, node)
        if (
            node.type
            in {"function_declaration", "method_definition", "generator_function_declaration"}
            and function_name != "<anonymous>"
        ):
            identifiers.append(function_name)  # fmt: skip
        functions.append(
            Function(
                name=function_name,
                line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                statements=_statements(node),
                has_docstring=_declaration_start(node) - 1 in jsdoc_end_rows,
            )
        )

    return SourceFile(
        path=path,
        language="typescript" if kind != "javascript" else "javascript",
        lines=lines,
        code_line_numbers={r + 1 for r in code_rows - docstring_rows},
        comments=comments,
        docstring_lines=len(docstring_rows),
        functions=functions,
        identifiers=identifiers,
        quote_styles=quote_styles,
        commented_code_lines=[
            c.line
            for c in comments
            if c.own_line and c.line_count == 1 and looks_like_javascript(c.text, kind)
        ],  # fmt: skip
        debug_statements=debug,
    )
