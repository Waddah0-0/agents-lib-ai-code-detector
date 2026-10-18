"""Source parsing and each file-level signal, in both directions."""

from __future__ import annotations

from pathlib import Path

from ai_code_detector.methodology import load_methodology
from ai_code_detector.signals.file import (
    CommentDensity,
    DeadOrDebugCode,
    Idiosyncrasy,
    NamingPatterns,
    RestatingComments,
)
from ai_code_detector.source import parse_source, split_words
from ai_code_detector.source.python import parse_python

M = load_methodology()
FIXTURES = Path(__file__).parent / "fixtures"
AI_STYLE_FILE = (FIXTURES / "ai_style.py.txt").read_text(encoding="utf-8")
HUMAN_STYLE_FILE = (FIXTURES / "human_style.py.txt").read_text(encoding="utf-8")


def parsed(text: str, path: str = "m.py"):  # type: ignore[no-untyped-def]
    source = parse_python(path, text)
    assert source is not None
    return source


def test_parse_basics() -> None:
    source = parsed(
        '"""Doc."""\nx = 1  # inline\n# own line\n\n\ndef f(a, b):\n    """Doc."""\n    return a\n'
    )
    assert source.docstring_lines == 2
    assert [c.own_line for c in source.comments] == [False, True]
    assert source.code_lines == 3  # x = 1, def, return
    assert source.functions[0].statements == 1 and source.functions[0].has_docstring
    assert {"x", "f", "a", "b"} <= set(source.identifiers)


def test_unparseable_is_none() -> None:
    assert parse_python("bad.py", "def (:\n") is None


def test_split_words() -> None:
    assert split_words("getUserName and parse_HTTPResponse") == [
        "get", "user", "name", "and", "parse", "http", "response",
    ]  # fmt: skip


def strengths(signal: object, text: str) -> list[float]:
    return [h.strength for h in signal.evaluate(parsed(text), M)]  # type: ignore[attr-defined]


def test_comment_density_both_ways() -> None:
    assert strengths(CommentDensity(), AI_STYLE_FILE)[0] > 0.3
    sparse = "\n".join(f"v{i} = {i} * 2" for i in range(70)) + "\n"
    assert strengths(CommentDensity(), sparse) == [-0.3]


def test_restating_comments() -> None:
    hits = RestatingComments().evaluate(parsed(AI_STYLE_FILE), M)
    assert hits and hits[0].strength > 0.5
    assert "restate" in hits[0].explanation
    assert strengths(RestatingComments(), HUMAN_STYLE_FILE) == []


def test_naming_patterns_both_ways() -> None:
    generic = "\n".join(
        f"def process_item_{i}(input_data, result_list):\n    output_data = input_data\n"
        f"    processed_data = output_data\n    return processed_data\n"
        for i in range(8)
    )
    assert strengths(NamingPatterns(), generic)[0] > 0
    assert strengths(NamingPatterns(), HUMAN_STYLE_FILE)[0] < 0


def test_idiosyncrasy_is_human_evidence() -> None:
    hits = Idiosyncrasy().evaluate(parsed(HUMAN_STYLE_FILE), M)
    assert hits and hits[0].strength < 0
    assert "personal note" in hits[0].evidence[0].detail
    assert strengths(Idiosyncrasy(), AI_STYLE_FILE) == []


def test_dead_and_debug_code_is_human_evidence() -> None:
    hits = DeadOrDebugCode().evaluate(parsed(HUMAN_STYLE_FILE), M)
    assert hits and hits[0].strength < 0
    assert "1 blocks of commented-out code" in hits[0].explanation
    assert "1 leftover debugging" in hits[0].explanation
    assert strengths(DeadOrDebugCode(), AI_STYLE_FILE) == []


def test_prose_comments_are_not_code() -> None:
    text = "# Note: this is fine.\n# see the docs for details\nx = 1\n"
    assert strengths(DeadOrDebugCode(), text) == []


TS_AI = (FIXTURES / "ai_style.ts.txt").read_text(encoding="utf-8")
JS_HUMAN = (FIXTURES / "human_style.js.txt").read_text(encoding="utf-8")


def test_parse_typescript() -> None:
    source = parse_source("m.ts", TS_AI)
    assert source is not None and source.language == "typescript"
    names = {f.name: f for f in source.functions}
    assert {"processData", "getValue", "isValid", "formatOutput", "filterItems"} <= set(names)
    assert names["isValid"].has_docstring and names["getValue"].has_docstring  # JSDoc above
    assert names["isValid"].statements == 1
    assert source.docstring_lines > 10
    assert "inputData" in source.identifiers and "resultList" in source.identifiers


def test_parse_javascript_human_traces() -> None:
    source = parse_source("m.js", JS_HUMAN)
    assert source is not None and source.language == "javascript"
    assert len(source.commented_code_lines) == 2
    assert [text for _, text in source.debug_statements] == ['console.log("here")', "debugger"]
    assert source.quote_styles["'"] and source.quote_styles['"']


def test_unsupported_and_broken_sources() -> None:
    assert parse_source("types.go", "package main") is None
    assert parse_source("broken.js", "function (( {{{ ]]] " * 50) is None


def test_typescript_signals_lean_ai() -> None:
    source = parse_source("m.ts", TS_AI)
    assert source is not None
    signals = (CommentDensity(), RestatingComments(), NamingPatterns())
    hits = {h.signal_id: h.strength for s in signals for h in s.evaluate(source, M)}
    assert hits["comment_density"] > 0.5
    assert hits["restating_comments"] > 0.5
    assert hits["naming_patterns"] > 0  # camelCase generic names: resultList, inputData


def test_javascript_signals_lean_human() -> None:
    source = parse_source("m.js", JS_HUMAN)
    assert source is not None
    hits = {h.signal_id: h.strength for s in (Idiosyncrasy(), DeadOrDebugCode(), NamingPatterns())
            for h in s.evaluate(source, M)}  # fmt: skip
    assert hits["idiosyncrasy"] < 0 and hits["dead_or_debug_code"] < 0
    assert hits["naming_patterns"] < 0
