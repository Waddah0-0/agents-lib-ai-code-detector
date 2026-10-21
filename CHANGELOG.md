# Changelog

All notable changes to this agent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-21

The first published version.

### Added
- Reads Python, JavaScript and TypeScript, and the repository's git history.
- A local model judges a sample of files and points at the lines behind its view, for and
  against. The quotes in the report are taken from the file, not from the model, and the code
  it reads is fenced off so instructions hidden in a repository can't steer it.
- Methodology v1: the judge's verdicts and the file-level checks are combined with weights
  fitted on 44 repositories. Commit history and README style are shown for context but not
  scored, because in testing they separated old projects from new ones rather than people
  from AI.
- Traces left by AI tools (config files such as `CLAUDE.md`, co-author trailers, bot authors)
  for nine tools, each checked against a real commit. They're reported on their own and
  never change the estimate.
- An estimate with a range and a confidence level, the evidence on both sides, a score per
  file, code excerpts around the evidence, and a commit history without author names.
- Without a model it still runs on the file checks alone, and says the confidence is low.

### Known limitations
- The scores aren't calibrated yet, so confidence never goes above medium.
- Tuned to avoid false accusations: at a setting that keeps false positives at 5% or less,
  it misses most AI-written repositories.
- Repositories with big generated-looking data files, such as locale tables, can score high.
- Not yet tested on code written with light AI help, or on AI code that people edited.

## [0.1.0] - not published

A development version with file checks only (methodology v0), used to build the evaluation
data. Never released.
