# Changelog

All notable changes to this agent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

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
