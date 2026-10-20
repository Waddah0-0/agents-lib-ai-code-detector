# Changelog

All notable changes to this agent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-21

### Added
- JavaScript and TypeScript file signals (tree-sitter).
- LLM judge (`judge_v2`): the model reads a sample of files and points at lines as evidence
  for and against; quotes are taken from the file. Code is fenced by random markers and
  treated as data. Cached, time- and cost-limited, with a self-agreement check.
- Methodology v1, fitted on dataset v1's train split: the judge carries most of the weight;
  commit-history and README signals are shown as context but not scored.
- Report data for UIs: code excerpts around the evidence, a commit list (without authors) and
  every unscored file with its reason. Additive; existing fields are unchanged.
- Dataset v1 (70 repositories, Python and JS/TS, six AI tools, recent human code) and the
  tools to fit and compare methodologies and judge models.

### Changed
- The judge runs by default when a model is configured; without one the detector falls back
  to a signals-only model and reports low confidence.
- Default local model: qwen3.5:9b (chosen by measurement), with hidden reasoning turned off.
- Progress reporting gives the judge (the slow part) most of the progress bar, and `target`
  carries an example value for forms.

### Fixed
- The range ignored how few files the judge read: under methodology v1 one judged file could
  give a range like "0.78-0.78". The judge term is now recomputed for every bootstrap
  resample, and the reported range is at least ±0.15 wide at low confidence and ±0.07 at
  medium. Estimates are unchanged.
- Judge failures are reported once per kind of error instead of once per file, with the
  server's own message, and the judge stops after three failed calls in a row (for example
  when the local model does not fit in memory).

### Known limitations
- Not calibrated (validation calibration error 0.27): confidence is capped at medium.
- At a threshold that keeps false positives at or below 5%, it catches very few AI
  repositories. Repositories with large generated-looking data files (e.g. locale tables)
  can score high.

## [0.1.0] - not published (development version)

### Added
- Heuristic detector (methodology v0, uncalibrated): repository signals `commit_shape`,
  `history_depth`, `readme_patterns`; Python file signals `comment_density`,
  `restating_comments`, `naming_patterns`, `idiosyncrasy`, `dead_or_debug_code`.
- Metadata markers (`markers/v1.toml`) for Claude, Claude Code, Gemini, Codex, Cursor, Aider,
  the Copilot coding agent, Devin and Jules, each verified against a real public commit.
  Reported separately; never part of the estimate.
- Estimate with a bootstrap range and a confidence level with reasons. Confidence is capped
  at "medium" while the methodology is uncalibrated.
- Evaluation harness (`eval/`) and dataset v0 with baseline results (`eval/results/v0.json`).
